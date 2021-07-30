// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// Local include(s).
#include "Acts/Plugins/Sycl/Seeding/detail/Types.hpp"

#include "../Utilities/Arrays.hpp"
#include "SpacePointType.hpp"

// SYCL include(s).
#include <CL/sycl.hpp>

// VecMem includes
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"

// System include(s).
#include <cstdint>

namespace Acts::Sycl::detail {

/// Functor taking care of finding viable spacepoint duplets
template <SpacePointType OtherSPType>
class DupletSearch {
  // Sanity check(s).
  static_assert((OtherSPType == SpacePointType::Bottom) ||
                    (OtherSPType == SpacePointType::Top),
                "Class must be instantiated with either "
                "Acts::Sycl::detail::SpacePointType::Bottom or "
                "Acts::Sycl::detail::SpacePointType::Top");

 public:
  /// Constructor with all the necessary arguments
  DupletSearch(const vecmem::data::vector_view<const detail::DeviceSpacePoint>& middleSPs,
               const vecmem::data::vector_view<const detail::DeviceSpacePoint>& otherSPs,
               vecmem::data::jagged_vector_view<uint32_t>& middleOtherSPIndicesView,
               const DeviceSeedfinderConfig& config)
      : m_nMiddleSPs(middleSPs.size()),
        m_middleSPs(middleSPs),
        m_nOtherSPs(otherSPs.size()),
        m_otherSPs(otherSPs),
        m_middleOtherSPIndicesView(middleOtherSPIndicesView),
        m_config(config) {}

  /// Operator performing the duplet search
  void operator()(cl::sycl::nd_item<2> item) const {
    // Get the indices of the spacepoints to evaluate.
    auto middleIndex = item.get_global_id(0);
    auto otherIndex = item.get_global_id(1);

    // We check whether this thread actually makes sense (within bounds).
    // The number of threads is usually a factor of 2, or 3*2^k (k \in N), etc.
    // Without this check we may index out of arrays.
    if ((middleIndex >= m_nMiddleSPs) || (otherIndex >= m_nOtherSPs)) {
      return;
    }

    // Creating device vecmem vectors needed.
    vecmem::device_vector<const detail::DeviceSpacePoint> device_middleSPs(m_middleSPs);
    vecmem::device_vector<const detail::DeviceSpacePoint> device_otherSPs(m_otherSPs);
    vecmem::jagged_device_vector<uint32_t> middleOtherSPIndices(m_middleOtherSPIndicesView);

    // Create a copy of the spacepoint objects for the current thread. On
    // dedicated GPUs this provides a better performance than accessing
    // variables one-by-one from global device memory.
    auto middleSP = device_middleSPs[middleIndex];
    auto otherSP = device_otherSPs[otherIndex];

    // Calculate the variables that the duplet quality selection are based on.
    // Note that the asserts of the functor make sure that 'OtherSPType' must be
    // either SpacePointType::Bottom or SpacePointType::Top.
    float deltaR = 0.0f, cotTheta = 0.0f;
    if constexpr (OtherSPType == SpacePointType::Bottom) {
      deltaR = middleSP.r - otherSP.r;
      cotTheta = (middleSP.z - otherSP.z) / deltaR;
    } else {
      deltaR = otherSP.r - middleSP.r;
      cotTheta = (otherSP.z - middleSP.z) / deltaR;
    }
    const float zOrigin = middleSP.z - middleSP.r * cotTheta;

    // Check if the duplet passes our quality requirements.
    if ((deltaR >= m_config.deltaRMin) && (deltaR <= m_config.deltaRMax) &&
        (cl::sycl::abs(cotTheta) <= m_config.cotThetaMax) &&
        (zOrigin >= m_config.collisionRegionMin) &&
        (zOrigin <= m_config.collisionRegionMax)) {
      middleOtherSPIndices[middleIndex].push_back(otherIndex);
    }
  }

 private:
  /// Total number of middle spacepoints
  uint32_t m_nMiddleSPs;
  /// VecMem vector view to the Middle space points
  const vecmem::data::vector_view<const detail::DeviceSpacePoint> m_middleSPs;
  /// Total number of "other" (bottom or top) spacepoints
  uint32_t m_nOtherSPs;
  /// VecMem vector view to the "other" (bottom or top) spacepoints (in global device mem.)
  const vecmem::data::vector_view<const detail::DeviceSpacePoint> m_otherSPs;

  /// The 2D array (jagged vector now) storing the compatible middle-other spacepoint indices
  vecmem::data::jagged_vector_view<uint32_t> m_middleOtherSPIndicesView;

  /// Configuration for the seed finding
  DeviceSeedfinderConfig m_config;

};  // struct DupletSearch

}  // namespace Acts::Sycl::detail
