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
std::array<RI::Tensor<Tdata>, 3> Matrix_Orbs21_r::cal_overlap_matrix(
    const size_t TA,
    const size_t TB,
    const ModuleBase::Vector3<double>& tauA,
    const ModuleBase::Vector3<double>& tauB,
    const ModuleBase::Element_Basis_Index::IndexLNM& index_A,
    const ModuleBase::Element_Basis_Index::IndexLNM& index_B,
    const Matrix_Orbs11::Matrix_Order& matrix_order)
{
    using namespace RI::Array_Operator;
    std::array<RI::Tensor<Tdata>, 3> m;
    const size_t sizeA = index_A[TA].count_size;
    const size_t sizeB = index_B[TB].count_size;
    for (size_t i = 0; i < m.size(); ++i)
    {
        switch (matrix_order)
        {
        case Matrix_Orbs11::Matrix_Order::AB:
            m[i] = RI::Tensor<Tdata>({sizeA, sizeB});
            break;
        case Matrix_Orbs11::Matrix_Order::BA:
            m[i] = RI::Tensor<Tdata>({sizeB, sizeA});
            break;
        default:
            throw std::invalid_argument(std::string(__FILE__) + " line " + std::to_string(__LINE__));
        }
    }
    ModuleBase::Vector3<double> origin_point(0.0, 0.0, 0.0);
    double factor = sqrt(ModuleBase::FOUR_PI / 3.0);
    const ModuleBase::Vector3<double>& distance = (tauB - tauA) * GlobalC::ucell.lat0;

    for (const auto& co3: this->center2_orb21_r.at(TA).at(TB))
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
                            std::array<Tdata, 3> overlap;
                            double overlap_o
                                = this->center2_orb11_r[TA][TB][LA][NA][LB].at(NB).cal_overlap(origin_point, distance, MA, MB);

                            overlap[0] = -1 * factor
                                         * co6.second.cal_overlap(origin_point, distance,
                                                                  MA,
                                                                  1,
                                                                  MB); // m =  1

                            overlap[1] = -1 * factor
                                         * co6.second.cal_overlap(origin_point, distance,
                                                                  MA,
                                                                  2,
                                                                  MB); // m = -1

                            overlap[2] = factor * co6.second.cal_overlap(origin_point, distance, MA, 0,
                                                                         MB); // m =  0
                            const size_t iA = index_A[TA][LA][NA][MA];
                            const size_t iB = index_B[TB][LB][NB][MB];
                            for (size_t i = 0; i < m.size(); ++i)
                            {
                                switch (matrix_order)
                                {
                                case Matrix_Orbs11::Matrix_Order::AB:
                                    m[i](iA, iB) = overlap[i] + tauA[i] * GlobalC::ucell.lat0 * overlap_o;
                                    break;
                                case Matrix_Orbs11::Matrix_Order::BA:
                                    m[i](iB, iA) = overlap[i] + tauA[i] * GlobalC::ucell.lat0 * overlap_o;
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