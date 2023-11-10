#ifndef EXX_ABFS_CONSTRUCT_ORBS_H
#define EXX_ABFS_CONSTRUCT_ORBS_H

#include "exx_abfs.h"

#include <limits>
#include "../module_basis/module_ao/ORB_atomic_lm.h"

	class LCAO_Orbitals;

class Exx_Abfs::Construct_Orbs
{
public:
	static std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> change_orbs( 
		const LCAO_Orbitals &orb_in,
		const double kmesh_times );
	static std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> change_orbs( 
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in,
		const double kmesh_times );

	static std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_same_atom( 
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos,
		const double kmesh_times_mot,
		const double times_threshold=0);
		
	static void print_orbs_size(
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs,
		std::ostream &os);		
	
	static int get_nmax_total(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in); // get the max number of orbitals among all elements
	static std::map<int, int> get_nw(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in); // get number of orbitals for each element 
	static int get_norb(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in); // get total number of orbitals
	static std::vector<int> get_iat2iwt(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in) // // iat ==> iwt, the first global index for orbital of this atom
	static inline int get_itiaiw2iwt(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in, const int &it, const int &ia, const int &iw) // indexing tool for find orbital global index from it,ia,iw 
	{
		return get_iat2iwt(orb_in)[GlobalC::ucell.itia2iat(it, ia)] + iw;
	}
		
private:
	static std::vector<std::vector<std::vector<std::vector<double>>>> psi_mult_psi( 
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos );
		
	static std::vector<std::vector<std::vector<std::vector<double>>>> psir_mult_psir( 
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos );
		
	static std::vector<std::vector<std::vector<std::vector<double>>>> orth( 
		const std::vector<std::vector<std::vector<std::vector<double>>>> &psis,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos,
		const double norm_threshold = std::numeric_limits<double>::min() );

	static std::vector<std::vector<std::vector<std::vector<double>>>> pca(
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &abfs,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs,
		const double kmesh_times_mot,
		const double times_threshold );
		
	static std::vector<std::vector<std::vector<std::vector<double>>>> div_r( 
		const std::vector<std::vector<std::vector<std::vector<double>>>> &psirs,
		const std::vector<double> &r_radial );		
		
	static std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> orbital(
		const std::vector<std::vector<std::vector<std::vector<double>>>> &psis,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs_info,
		const double kmesh_times);		
		
	static std::vector<std::vector<std::vector<std::vector<double>>>> get_psi(
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs );
};

#endif	// EXX_ABFS_IO_ASA_H
