
/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <typeinfo>
using host_memory_space = Kokkos::Experimental::FakeHostSpace;
using cuda_memory_space = Kokkos::Experimental::FakeGPUSpace;
using host_execution_space = Kokkos::Serial;
using device_execution_space = Kokkos::Experimental::FakeGPU;

namespace Kokkos {
namespace Impl {
template<> struct MemorySpaceAccess <Kokkos::HostSpace, Kokkos::Experimental::FakeGPUSpace> {
  enum { accessible = true };
  enum { assignable = true };
  enum { deepcopy = true };
};
template<> struct MemorySpaceAccess <Kokkos::HostSpace, Kokkos::Experimental::FakeHostSpace> {
  enum { accessible = true };
  enum { assignable = true };
  enum { deepcopy = true };
};


//template<class HV, class DV>
//class ViewMapping <HV,DV, FakeSpaceTag> {
//    public:
//    enum : bool { is_assignable_data_type = true };
//    enum : bool { is_assignable = true }; 
//};

template<class, class>
struct CopyHandle;
template<class Space, class... Props>
struct CopyHandle<Space, Kokkos::View<Props...>> {
  using view = Kokkos::View<Props...>;
  using type = decltype(Kokkos::create_mirror_view_and_copy(std::declval<Space>(),std::declval<view>()));
  //static type get(const view& in, const Space& space = Space()){
  //  auto ret = Kokkos::create_mirror_view_and_copy(space, in);
  //  return ret;
  //}
  static type get(const view& in, const Space& space = Space()){
    auto ret = Kokkos::create_mirror_view(space, in);
    ret.assign_data(in.data());
    return ret;
  }
};

template<class Space, class Scalar>
struct CopyHandle{
  using type = const Scalar&;
  static type get(type& in, const Space& space = Space()) {
    return in;
  }
};
template<class Copied, class Space = Kokkos::HostSpace>
auto get_copy_handle(const Copied& cop, const Space& spac = Space()){
  return CopyHandle<Space, Copied>::get(cop, spac);
}



} // namespace Impl


template<class D, class S>
inline void deep_copy(
  const D& dst,
  const S& src,
  typename std::enable_if<
    (!Impl::not_logical_view<D>::value) ||
    (!Impl::not_logical_view<S>::value),
    void
    >::type* = nullptr
){

  using Dref = const typename Impl::LogicalToBase<D>::type;
  using Sref = const typename Impl::LogicalToBase<S>::type;
  Kokkos::HostSpace hs;
  //Dref d(dst.data());
  //Sref s(src.data());
  
  auto dm = Impl::get_copy_handle(dst, hs);
  auto sm = Impl::get_copy_handle(src, hs);
  static_assert(Impl::not_logical_view<decltype(dm)>::value, "dm is still an lv");
  static_assert(Impl::not_logical_view<decltype(sm)>::value, "sm is still an lv");
  deep_copy(dm,sm);
}

} // namespace Kokkos

int main(){
Kokkos::initialize();
{
	/**
	Kokkos::View<float*,host_memory_space> test_host_view("host_view",1000);
	Kokkos::View<float*,cuda_memory_space> test_device_view("device_view",1000);
	Kokkos::View<float*,Kokkos::HostSpace> test_rh_view("rh_view",1000);
	
Kokkos::parallel_for("dogs",Kokkos::RangePolicy<device_execution_space>(0,800), KOKKOS_LAMBDA(int i){
  test_device_view(i) = i;
		});
        //Kokkos::View<float*, Kokkos::HostSpace> th(test_host_view); 
        Kokkos::deep_copy(test_host_view, test_device_view);
        Kokkos::deep_copy(test_device_view, test_host_view);
        Kokkos::deep_copy(test_device_view, test_rh_view);
        Kokkos::deep_copy(test_host_view, test_rh_view);
        Kokkos::deep_copy(test_rh_view, test_device_view);
        Kokkos::deep_copy(test_rh_view, test_host_view);
        Kokkos::deep_copy(test_host_view, 1.0f);
        Kokkos::deep_copy(test_device_view, 1.0f);
	*/
	Kokkos::DualView<float*, cuda_memory_space> dv("puppies",1000);
        //dv.h_view(5);
        //dv.d_view(5);
	std::cout << dv.h_view.data() << ", "<<dv.d_view.data()<<std::endl;
        }
	Kokkos::finalize();
}

