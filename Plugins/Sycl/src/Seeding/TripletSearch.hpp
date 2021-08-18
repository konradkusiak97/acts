
#pragma once

// Local include(s).
#include "Acts/Plugins/Sycl/Seeding/detail/Types.hpp"

#include "../Utilities/Arrays.hpp"
#include "SpacePointType.hpp"

// VecMem include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"

// SYCL include(s).
#include <CL/sycl.hpp>

// System include(s).
#include <cstdint>

namespace Acts::Sycl::detail {

/// Functor performing Triplet Search
template <class AtomicAccessorType>
class TripletSearch {

    public:
    /// Constructor
    TripletSearch(vecmem::data::vector_view<uint32_t> sumBotTopCombView,
                  const uint32_t numTripletSearchThreads,
                  const uint32_t firstMiddle,
                  const uint32_t lastMiddle,
                  vecmem::data::jagged_vector_view<const uint32_t> midTopDupletView,
                  vecmem::data::vector_view<uint32_t> sumBotMidView,
                  vecmem::data::vector_view<uint32_t> sumTopMidView,
                  vecmem::data::vector_view<detail::DeviceLinEqCircle> linearBotView,
                  vecmem::data::vector_view<detail::DeviceLinEqCircle> linearTopView,
                  vecmem::data::vector_view<const detail::DeviceSpacePoint> middleSPsView,
                  vecmem::data::vector_view<uint32_t> indTopDupletview,
                  const AtomicAccessorType& countTripletsAcc,
                  const DeviceSeedfinderConfig& config,
                  vecmem::data::vector_view<detail::DeviceTriplet> curvImpactView)
          : m_sumBotTopCombView(sumBotTopCombView),
            m_numTripletSearchThreads(numTripletSearchThreads),
            m_firstMiddle(firstMiddle),
            m_lastMiddle(lastMiddle),
            m_midTopDupletView(midTopDupletView),
            m_sumBotMidView(sumBotMidView),
            m_sumTopMidView(sumTopMidView),
            m_linearBotView(linearBotView),
            m_linearTopView(linearTopView),
            m_middleSPsView(middleSPsView),
            m_indTopDupletView(indTopDupletview),
            m_countTripletsAcc(countTripletsAcc),
            m_config(config),
            m_curvImpactView(curvImpactView) {}

    /// Operator performing the triplet search
    void operator()(cl::sycl::nd_item<1> item) const {
        // Get the index
        const uint32_t idx = item.get_global_linear_id();
        if (idx < m_numTripletSearchThreads) {
            // Retrieve the index of the corresponding middle
            // space point by binary search
            vecmem::device_vector<uint32_t>
                     sumBotTopCombPrefix(m_sumBotTopCombView);
            const auto sumCombUptoFirstMiddle = 
                     sumBotTopCombPrefix[m_firstMiddle];
            auto L = m_firstMiddle;
            auto R = m_lastMiddle;
            auto mid = L;
            while (L < R - 1) {
                mid = (L + R) / 2;
                // To be able to search in sumBotTopCombPrefix, we need
                // to use an offset (sumCombUptoFirstMiddle).
                if (idx + sumCombUptoFirstMiddle < sumBotTopCombPrefix[mid]) {
                    R = mid;
                } else {
                    L = mid;
                }
            }
            mid = L;
            vecmem::jagged_device_vector<const uint32_t>
                    midTopDuplets(m_midTopDupletView);
            const auto numT = midTopDuplets.at(mid).size();
            const auto threadIdxForMiddleSP =
                (idx - sumBotTopCombPrefix[mid] + sumCombUptoFirstMiddle);

            vecmem::device_vector<uint32_t>
                      sumBotMidPrefix(m_sumBotMidView);
            const auto ib =
                sumBotMidPrefix[mid] + (threadIdxForMiddleSP / numT);
            vecmem::device_vector<uint32_t>
                sumTopMidPrefix(m_sumTopMidView);
            const auto it =
                sumTopMidPrefix[mid] + (threadIdxForMiddleSP % numT);
            vecmem::device_vector<detail::DeviceLinEqCircle>
                                    deviceLinBot(m_linearBotView);
            const auto linBotEq = deviceLinBot[ib];
            vecmem::device_vector<detail::DeviceLinEqCircle>
                                    deviceLinTop(m_linearTopView);
            const auto linTopEq = deviceLinTop[it];
            const vecmem::device_vector<const detail::DeviceSpacePoint>
                                        middleSPs(m_middleSPsView);
            const auto midSP = middleSPs[mid];

            const auto Vb = linBotEq.v;
            const auto Ub = linBotEq.u;
            const auto Erb = linBotEq.er;
            const auto cotThetab = linBotEq.cotTheta;
            const auto iDeltaRb = linBotEq.iDeltaR;

            const auto Vt = linTopEq.v;
            const auto Ut = linTopEq.u;
            const auto Ert = linTopEq.er;
            const auto cotThetat = linTopEq.cotTheta;
            const auto iDeltaRt = linTopEq.iDeltaR;

            const auto rM = midSP.r;
            const auto varianceRM = midSP.varR;
            const auto varianceZM = midSP.varZ;

            auto iSinTheta2 = (1.f + cotThetab * cotThetab);
            auto scatteringInRegion2 =
                m_config.maxScatteringAngle2 * iSinTheta2;
            scatteringInRegion2 *= m_config.sigmaScattering *
                                    m_config.sigmaScattering;
            auto error2 =
                Ert + Erb +
                2.f * (cotThetab * cotThetat * varianceRM + varianceZM) *
                    iDeltaRb * iDeltaRt;
            auto deltaCotTheta = cotThetab - cotThetat;
            auto deltaCotTheta2 = deltaCotTheta * deltaCotTheta;

            deltaCotTheta = cl::sycl::abs(deltaCotTheta);
            auto error = cl::sycl::sqrt(error2);
            auto dCotThetaMinusError2 =
                deltaCotTheta2 + error2 - 2.f * deltaCotTheta * error;
            auto dU = Ut - Ub;

            if ((!(deltaCotTheta2 - error2 > 0.f) ||
                 !(dCotThetaMinusError2 > scatteringInRegion2)) &&
                !(dU == 0.f)) {
              auto A = (Vt - Vb) / dU;
              auto S2 = 1.f + A * A;
              auto B = Vb - A * Ub;
              auto B2 = B * B;

              auto iHelixDiameter2 = B2 / S2;
              auto pT2scatter =
              4.f * iHelixDiameter2 * m_config.pT2perRadius;
              auto p2scatter = pT2scatter * iSinTheta2;
              auto Im = cl::sycl::abs((A - B * rM) * rM);

              if (!(S2 < B2 * m_config.minHelixDiameter2) &&
                  !((deltaCotTheta2 - error2 > 0.f) &&
                    (dCotThetaMinusError2 >
                     p2scatter * m_config.sigmaScattering *
                     m_config.sigmaScattering)) &&
                    !(Im > m_config.impactMax)) {
                    vecmem::device_vector<uint32_t> 
                        deviceIndTopDuplets(m_indTopDupletView);
                    const auto top = deviceIndTopDuplets[it];
                    // this will be the t-th top space point for
                    // fixed middle and bottom SP
                    auto t = m_countTripletsAcc[ib].fetch_add(1);

                    /* 
                        sumBotTopCombPrefix[mid] - sumCombUptoFirstMiddle:
                        gives the memory location reserved for this
                        middle SP

                        (idx-sumBotTopCombPrefix[mid]+sumCombUptoFirstMiddle:
                        this is the nth thread for this middle SP

                        (idx-sumBotTopCombPrefix[mid]+sumCombUptoFirstMiddle)/numT:
                        this is the mth bottom SP for this middle SP

                        multiplying this by numT gives the memory
                        location for this middle and bottom SP

                        and by adding t to it, we will end up storing
                        compatible triplet candidates for this middle
                        and bottom SP right next to each other
                        starting from the given memory 
                    */
                    const auto tripletIdx = sumBotTopCombPrefix[mid] -
                                              sumCombUptoFirstMiddle +
                                              (((idx - sumBotTopCombPrefix[mid] +
                                                 sumCombUptoFirstMiddle) /
                                                numT) *
                                               numT) +
                                              t;

                      detail::DeviceTriplet T;
                      T.curvature = B / cl::sycl::sqrt(S2);
                      T.impact = Im;
                      T.topSPIndex = top;
                      vecmem::device_vector
                          <detail::DeviceTriplet>
                              deviceCurvImpact(m_curvImpactView);
                      deviceCurvImpact[tripletIdx] = T;
             }
        }
      }
    }
    private:
        vecmem::data::vector_view<uint32_t> m_sumBotTopCombView;
        const uint32_t m_numTripletSearchThreads;
        const uint32_t m_firstMiddle;
        const u_int32_t m_lastMiddle;
        vecmem::data::jagged_vector_view<const uint32_t> m_midTopDupletView;
        vecmem::data::vector_view<uint32_t> m_sumBotMidView;
        vecmem::data::vector_view<uint32_t> m_sumTopMidView;
        vecmem::data::vector_view<detail::DeviceLinEqCircle> m_linearBotView;
        vecmem::data::vector_view<detail::DeviceLinEqCircle> m_linearTopView;
        vecmem::data::vector_view<const detail::DeviceSpacePoint> m_middleSPsView;
        vecmem::data::vector_view<uint32_t> m_indTopDupletView;
        AtomicAccessorType m_countTripletsAcc;
        DeviceSeedfinderConfig m_config;
        vecmem::data::vector_view<detail::DeviceTriplet> m_curvImpactView;
};  // struct TripletSearch

} // namespace Acts::Sycl::detail