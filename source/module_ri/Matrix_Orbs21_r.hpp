//=======================
// AUTHOR : Yuyang Ji
// DATE :   2025-02-23
//=======================

#ifndef MATRIX_ORB21_r_HPP
#define MATRIX_ORB21_r_HPP

#include "Matrix_Orbs21.h"
#include "RI_Util.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

template <typename Tdata>
std::array<RI::Tensor<Tdata>, 3> Matrix_Orbs21_r::cal_overlap_matrix(const size_t TA,
                                                    const size_t TB,
                                                    const ModuleBase::Vector3<double>& tauA,
                                                    const ModuleBase::Vector3<double>& tauB,
                                                    const ModuleBase::Element_Basis_Index::IndexLNM& index_A,
                                                    const ModuleBase::Element_Basis_Index::IndexLNM& index_B,
                                                    const Matrix_Order& matrix_order) const
{
    std::array<RI::Tensor<Tdata>, 3> m;
    const size_t sizeA = index_A[TA].count_size;
    const size_t sizeB = index_B[TB].count_size;
    switch (matrix_order)
    {
    case Matrix_Order::AB:
        m = RI::Tensor<Tdata>({sizeA, sizeB});
        break;
    case Matrix_Order::BA:
        m = RI::Tensor<Tdata>({sizeB, sizeA});
        break;
    default:
        throw std::invalid_argument(std::string(__FILE__) + " line " + std::to_string(__LINE__));
    }
    double factor = sqrt(ModuleBase::FOUR_PI / 3.0);

    for (const auto& co3: this->center2_orb11.at(TA).at(TB))
    {
        const int LA = co3.first;
        for (const auto& co4: co3.second)
        {
            const size_t NA = co4.first;
            for (size_t MA = 0; MA != 2 * LA + 1; ++MA)
            {
                for (const auto& co5: co4.second)
                {
                    const int LB = co5.first;
                    for (const auto& co6: co5.second)
                    {
                        const size_t NB = co6.first;
                        for (size_t MB = 0; MB != 2 * LB + 1; ++MB)
                        {

                            const Tdata overlap_o = co6.second.cal_overlap(tauA*GlobalC::ucell.lat0,
                                                                           tauB*GlobalC::ucell.lat0,
                                                                           MA,
                                                                           MB);
                            double overlap_x = -1 * factor
                                               * this->center2_orb21_r[TA][TB][LA][NA][LB].at(NB).cal_overlap(tauA*GlobalC::ucell.lat0,
                                                                                                        tauB*GlobalC::ucell.lat0,
                                                                                                        MA,
                                                                                                        1,
                                                                                                        MB); // m =  1

                            double overlap_y = -1 * factor
                                               * this->center2_orb21_r[TA][TB][LA][NA][LB].at(NB).cal_overlap(tauA*GlobalC::ucell.lat0,
                                                                                                        tauB*GlobalC::ucell.lat0,
                                                                                                        MA,
                                                                                                        2,
                                                                                                        MB); // m = -1

                            double overlap_z = factor
                                               * this->center2_orb21_r[TA][TB][LA][NA][LB].at(NB).cal_overlap(tauA*GlobalC::ucell.lat0,
                                                                                                        tauB*GlobalC::ucell.lat0,
                                                                                                        MA,
                                                                                                        0,
                                                                                                        MB); // m =  0
                            const size_t iA = index_A[TA][LA][NA][MA];
                            const size_t iB = index_B[TB][LB][NB][MB];
                            const ModuleBase::Vector3<double> overlap = ModuleBase::Vector3<double>(overlap_x, overlap_y, overlap_z) + tauA * overlap_o * GlobalC::ucell.lat0;
                            for(size_t i=0; i<m.size(); ++i)
							{
                                switch (matrix_order)
                                {
                                case Matrix_Order::AB:
                                    m[i](iA, iB) = overlap[i];
                                    break;
                                case Matrix_Order::BA:
                                    m[i](iB, iA) = overlap[i];
                                    break;
                                default:
                                    throw std::invalid_argument(std::string(__FILE__) + " line "
                                                                + std::to_string(__LINE__));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return m;
}
#endif