// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// SYCL plugin include(s).
#include "Acts/Plugins/Sycl/Seeding/detail/Types.hpp"

#include "../Utilities/Arrays.hpp"
#include "SpacePointType.hpp"

// SYCL include(s).
#include <CL/sycl.hpp>

// VecMem includes
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"

// System include(s).
#include <cassert>
#include <cstdint>

namespace Acts::Sycl::detail {

/// Functor performing a linear coordinate transformation on spacepoint pairs
template <SpacePointType OtherSPType>
class LinearTransform {
  // Sanity check(s).
  static_assert((OtherSPType == SpacePointType::Bottom) ||
                    (OtherSPType == SpacePointType::Top),
                "Class must be instantiated with either "
                "Acts::Sycl::detail::SpacePointType::Bottom or "
                "Acts::Sycl::detail::SpacePointType::Top");

 public:
  /// Constructor with all the necessary arguments
  LinearTransform(const vecmem::data::vector_view<const detail::DeviceSpacePoint>& middleSPs,
                  const vecmem::data::vector_view<const detail::DeviceSpacePoint>& otherSPs,
                  const vecmem::data::vector_view<u_int32_t>& middleIndexLUT,
                  const vecmem::data::vector_view<u_int32_t>& otherIndexLUT, uint32_t nEdges,
                  vecmem::data::vector_view<DeviceLinEqCircle>& resultArray)
      : m_nMiddleSPs(middleSPs.size()),
        m_middleSPs(middleSPs),
        m_nOtherSPs(otherSPs.size()),
        m_otherSPs(otherSPs),
        m_middleIndexLUT(middleIndexLUT),
        m_otherIndexLUT(otherIndexLUT),
        m_nEdges(nEdges),
        m_resultArray(resultArray) {}

  /// Operator performing the coordinate linear transformation
  void operator()(cl::sycl::nd_item<1> item) const {
    // Get the index to operate on.
    const auto idx = item.get_global_linear_id();
    if (idx >= m_nEdges) {
      return;
    }

    // Create the device VecMem vectors from the views.
    vecmem::device_vector<const detail::DeviceSpacePoint> device_middleSPs(m_middleSPs);
    vecmem::device_vector<const detail::DeviceSpacePoint> device_otherSPs(m_otherSPs);
    vecmem::device_vector<uint32_t> device_middleIndexLUT(m_middleIndexLUT);
    vecmem::device_vector<uint32_t> device_otherIndexLUT(m_otherIndexLUT);
    vecmem::device_vector<DeviceLinEqCircle> device_resultArray(m_resultArray);

    // Translate this one index into indices in the spacepoint arrays.
    // Note that using asserts with the CUDA backend of dpc++ is not working
    // quite correctly at the moment. :-( So these checks may need to be
    // disabled if you need to build for an NVidia backend in Debug mode.
    const uint32_t middleIndex = device_middleIndexLUT[idx];
    assert(middleIndex < m_nMiddleSPs);
    (void)m_nMiddleSPs;
    const uint32_t otherIndex = device_otherIndexLUT[idx];
    assert(otherIndex < m_nOtherSPs);
    (void)m_nOtherSPs;

    // Create a copy of the spacepoint objects for the current thread. On
    // dedicated GPUs this provides a better performance than accessing
    // variables one-by-one from global device memory.
    const DeviceSpacePoint middleSP = device_middleSPs[middleIndex];
    const DeviceSpacePoint otherSP = device_otherSPs[otherIndex];

    // Calculate some "helper variables" for the coordinate linear
    // transformation.
    const float cosPhiM = middleSP.x / middleSP.r;
    const float sinPhiM = middleSP.y / middleSP.r;

    const float deltaX = otherSP.x - middleSP.x;
    const float deltaY = otherSP.y - middleSP.y;
    const float deltaZ = otherSP.z - middleSP.z;

    const float x = deltaX * cosPhiM + deltaY * sinPhiM;
    const float y = deltaY * cosPhiM - deltaX * sinPhiM;
    const float iDeltaR2 = 1.f / (deltaX * deltaX + deltaY * deltaY);

    // Create the result object.
    DeviceLinEqCircle result;
    result.iDeltaR = cl::sycl::sqrt(iDeltaR2);
    result.cotTheta = deltaZ * result.iDeltaR;
    if constexpr (OtherSPType == SpacePointType::Bottom) {
      result.cotTheta = -(result.cotTheta);
    }
    result.zo = middleSP.z - middleSP.r * result.cotTheta;
    result.u = x * iDeltaR2;
    result.v = y * iDeltaR2;
    result.er =
        ((middleSP.varZ + otherSP.varZ) +
         (result.cotTheta * result.cotTheta) * (middleSP.varR + otherSP.varR)) *
        iDeltaR2;

    // Store the result object in device global memory.
    device_resultArray.push_back(result);
    return;
  }

 private:
  /// Total number of middle spacepoints
  uint32_t m_nMiddleSPs;
  /// VecMem vector view to the middle spacepoints (in global device memory)
  const vecmem::data::vector_view<const detail::DeviceSpacePoint> m_middleSPs;
  /// Total number of "other" (bottom or top) spacepoints
  uint32_t m_nOtherSPs;
  /// VecMem vector view to the "other" (bottom or top) spacepoints (in global device mem.)
  const vecmem::data::vector_view<const detail::DeviceSpacePoint> m_otherSPs;

  /// Look-Up Table from the iteration index to the middle spacepoint index
  const vecmem::data::vector_view<u_int32_t> m_middleIndexLUT;
  /// Loop-Up Table from the iteration index to the "other" spacepoint index
  const vecmem::data::vector_view<u_int32_t> m_otherIndexLUT;

  /// Total number of elements in the result array
  uint32_t m_nEdges;

  /// The result array in device global memory
  vecmem::data::vector_view<DeviceLinEqCircle> m_resultArray;

};  // class LinearTransform

}  // namespace Acts::Sycl::detail
