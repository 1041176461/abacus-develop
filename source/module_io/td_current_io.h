#ifndef W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_IO_TD_CURRENT_IO_H
#define W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_IO_TD_CURRENT_IO_H

#include "module_basis/module_nao/two_center_bundle.h"
#include "module_elecstate/elecstate_lcao.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_psi/psi.h"
#include "module_hamilt_lcao/module_tddft/td_current.h"

namespace ModuleIO
{
#ifdef __LCAO
/// @brief func to output current, only used in tddft
void write_current_eachk(const int istep,
                        const psi::Psi<std::complex<double>>* psi,
                        const elecstate::ElecState* pelec,
                        const K_Vectors& kv,
                        const TwoCenterIntegrator* intor,
                        const Parallel_Orbitals* pv,
                        const LCAO_Orbitals& orb,
                        const TD_current* cal_current,
                        Record_adj& ra,
#ifdef __EXX
                Exx_LRI<std::complex<double>>& exx_lri
#endif
                        );

void write_current(const int istep,
                const psi::Psi<std::complex<double>>* psi,
                const elecstate::ElecState* pelec,
                const K_Vectors& kv,
                const TwoCenterIntegrator* intor,
                const Parallel_Orbitals* pv,
                const LCAO_Orbitals& orb,
                const TD_current* cal_current,
                Record_adj& ra,
#ifdef __EXX
                Exx_LRI<std::complex<double>>& exx_lri
#endif
                );

/// @brief calculate sum_n[𝜌_(𝑛𝑘,𝜇𝜈)] for current calculation
void cal_tmp_DM_k(elecstate::DensityMatrix<std::complex<double>, double>& DM_real,
                elecstate::DensityMatrix<std::complex<double>, double>& DM_imag,
                const int ik,
                const int nspin,
                const int is,
                const bool reset = true);

void cal_tmp_DM(elecstate::DensityMatrix<std::complex<double>, double>& DM_real,
                elecstate::DensityMatrix<std::complex<double>, double>& DM_imag,
                const int nspin);

/// @brief cal [r, Hexx] for current density
void cal_current_exx(
    Exx_LRI<std::complex<double>>& exx,
    const elecstate::DensityMatrix<std::complex<double>, double>& dm,
    const K_Vectors& kv,
    const Parallel_Orbitals* pv,
    std::vector<hamilt::HContainer<std::complex<double>>*>& hR
);

#endif // __LCAO
} // namespace ModuleIO
#endif // W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_IO_TD_CURRENT_IO_H
