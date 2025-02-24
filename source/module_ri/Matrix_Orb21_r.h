//=======================
// AUTHOR : Yuyang Ji
// DATE :   2025-02-23
//=======================

#ifndef MATRIX_ORB21_R_H
#define MATRIX_ORB21_R_H

#include "module_base/element_basis_index.h"
#include "module_base/vector3.h"
#include "module_basis/module_ao/ORB_gaunt_table.h"
#include "module_basis/module_ao/ORB_read.h"
#include "module_hamilt_lcao/hamilt_lcaodft/center2_orb-orb21.h"

#include <RI/global/Tensor.h>
#include <map>
#include <set>
#include <vector>

class Matrix_Orbs21_r
{
  public:
    // mode:
    // 1: <lcaos|r|lcaos>
    // 2: <<jYs|r|jYs>  <abfs|r|abfs>>
    void init(const int mode,
              const LCAO_Orbitals& orb,
              const double kmesh_times, // extend Kcut, keep dK
              const double rmax,
              int Lmax); // extend Rcut, keep dR

    void init_radial(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_A,
                     const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_B,
                     const ORB_gaunt_table& MGT);
    void init_radial_table();

    template <typename Tdata>
    std::array<RI::Tensor<Tdata>, 3> cal_overlap_matrix(const size_t TA,
                                         const size_t TB,
                                         const ModuleBase::Vector3<double>& tauA,
                                         const ModuleBase::Vector3<double>& tauB,
                                         const ModuleBase::Element_Basis_Index::IndexLNM& index_A,
                                         const ModuleBase::Element_Basis_Index::IndexLNM& index_B,
                                         const Matrix_Order& matrix_order) const;

  private:
    ModuleBase::Sph_Bessel_Recursive::D2* psb_ = nullptr;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> orbs;
    Numerical_Orbital_Lm orb_r;
    std::map<size_t,                                              // TA
             std::map<size_t,                                     // TB
                      std::map<int,                               // LA
                               std::map<size_t,                   // NA
                                        std::map<int,             // LB
                                                 std::map<size_t, // NB
                                                          Center2_Orb::Orb11>>>>>>
        center2_orb11;

    std::map<size_t,                                                                // TA
             std::map<size_t,                                                       // TB
                      std::map<int,                                                 // LA1
                               std::map<size_t,                                     // NA1
                                        std::map<int,                               // LA2
                                                 std::map<size_t,                   // NA2
                                                          std::map<int,             // LB
                                                                   std::map<size_t, // NB
                                                                            Center2_Orb::Orb21>>>>>>>>
        center2_orb21_r;
}