/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/gauge/GaugeImpl.h

Copyright (C) 2015

Author: paboyle <paboyle@ph.ed.ac.uk>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */
#ifndef GRID_GAUGE_IMPL_TYPES_H
#define GRID_GAUGE_IMPL_TYPES_H


NAMESPACE_BEGIN(Grid);

#define CPS_MD_TIME

#ifdef CPS_MD_TIME
#define HMC_MOMENTUM_DENOMINATOR (2.0)
#else
#define HMC_MOMENTUM_DENOMINATOR (1.0)
#endif

////////////////////////////////////////////////////////////////////////
// Implementation dependent gauge types
////////////////////////////////////////////////////////////////////////

#define INHERIT_GIMPL_TYPES(GImpl)                  \
  typedef typename GImpl::Simd Simd;                \
  typedef typename GImpl::Scalar Scalar;	    \
  typedef typename GImpl::LinkField GaugeLinkField; \
  typedef typename GImpl::Field GaugeField;         \
  typedef typename GImpl::ComplexField ComplexField;\
  typedef typename GImpl::SiteField SiteGaugeField; \
  typedef typename GImpl::SiteComplex SiteComplex;  \
  typedef typename GImpl::SiteLink SiteGaugeLink;

#define INHERIT_FIELD_TYPES(Impl)		    \
  typedef typename Impl::Simd Simd;		    \
  typedef typename Impl::ComplexField ComplexField; \
  typedef typename Impl::SiteField SiteField;	    \
  typedef typename Impl::Field Field;

// hardcodes the exponential approximation in the template
//template <class S, int Nrepresentation = Nc, int Nexp = 12 > class GaugeImplTypes {
template <class S, int Nrepresentation = Nc, int Nexp = 12, bool isSp2n = false > class GaugeImplTypes {
public:
  typedef S Simd;
  typedef typename Simd::scalar_type scalar_type;
  typedef scalar_type Scalar;
  template <typename vtype> using iImplScalar     = iScalar<iScalar<iScalar<vtype> > >;
  template <typename vtype> using iImplGaugeLink  = iScalar<iScalar<iMatrix<vtype, Nrepresentation> > >;
  template <typename vtype> using iImplGaugeField = iVector<iScalar<iMatrix<vtype, Nrepresentation> >, Nd>;


  typedef iImplScalar<Simd>     SiteComplex;
  typedef iImplGaugeLink<Simd>  SiteLink;
  typedef iImplGaugeField<Simd> SiteField;

  typedef Lattice<SiteComplex> ComplexField;
  typedef Lattice<SiteLink>    LinkField; 
  typedef Lattice<SiteField>   Field;

  typedef SU<Nrepresentation> Group;

  // Guido: we can probably separate the types from the HMC functions
  // this will create 2 kind of implementations
  // probably confusing the users
  // Now keeping only one class


  // Move this elsewhere? FIXME
  static inline void AddLink(Field &U, LinkField &W, int mu) { // U[mu] += W
    autoView(U_v,U,AcceleratorWrite);
    autoView(W_v,W,AcceleratorRead);
    accelerator_for( ss, U.Grid()->oSites(), 1, {
      U_v[ss](mu) = U_v[ss](mu) + W_v[ss]();
    });
  }

  ///////////////////////////////////////////////////////////
  // Move these to another class
  // HMC auxiliary functions
  static inline void generate_momenta(Field &P, GridSerialRNG & sRNG, GridParallelRNG &pRNG) 
  {
    // Zbigniew Srocinsky thesis:
    //
    // P(p) =  N \Prod_{x\mu}e^-{1/2 Tr (p^2_mux)}
    // 
    // p_x,mu = c_x,mu,a T_a
    //
    // Tr p^2 =  sum_a,x,mu 1/2 (c_x,mu,a)^2
    //
    // Which implies P(p) =  N \Prod_{x,\mu,a} e^-{1/4 c_xmua^2  }
    //
    //                    =  N \Prod_{x,\mu,a} e^-{1/2 (c_xmua/sqrt{2})^2  }
    // 
    //
    // Expect cxmua variance sqrt(2).
    //
    // Must scale the momentum by sqrt(2) to invoke CPS and UKQCD conventions
    //
    LinkField Pmu(P.Grid());
    Pmu = Zero();
<<<<<<< HEAD

=======
>>>>>>> 470d4dcc6db9fa41faceac54908bcd60a08bfd8a
    for (int mu = 0; mu < Nd; mu++)
    {
        if (isSp2n == true)
        {
            const int nSp = Nrepresentation/2;
            Sp<nSp>::GaussianFundamentalLieAlgebraMatrix(pRNG, Pmu);
        } else
        {
        
<<<<<<< HEAD
            Group::GaussianFundamentalLieAlgebraMatrix(pRNG, Pmu);
        }

=======
            SU<Nrepresentation>::GaussianFundamentalLieAlgebraMatrix(pRNG, Pmu);
        }
>>>>>>> 470d4dcc6db9fa41faceac54908bcd60a08bfd8a
      RealD scale = ::sqrt(HMC_MOMENTUM_DENOMINATOR) ;
      Pmu = Pmu*scale;
      PokeIndex<LorentzIndex>(P, Pmu, mu);
    }
  }

  static inline Field projectForce(Field &P) { return Ta(P); }

  static inline void update_field(Field& P, Field& U, double ep){
    //static std::chrono::duration<double> diff;

    //auto start = std::chrono::high_resolution_clock::now();
    autoView(U_v,U,AcceleratorWrite);
    autoView(P_v,P,AcceleratorRead);
    accelerator_for(ss, P.Grid()->oSites(),1,{
      for (int mu = 0; mu < Nd; mu++) {
          if (isSp2n == true)
          {
              U_v[ss](mu) = ProjectOnSpGroup(Exponentiate(P_v[ss](mu), ep, Nexp) * U_v[ss](mu));
          } else
          {
              U_v[ss](mu) = ProjectOnGroup(Exponentiate(P_v[ss](mu), ep, Nexp) * U_v[ss](mu));
          }
        
      }
    });
   //auto end = std::chrono::high_resolution_clock::now();
   // diff += end - start;
   // std::cout << "Time to exponentiate matrix " << diff.count() << " s\n";
  }
    
  static inline RealD FieldSquareNorm(Field& U){
    LatticeComplex Hloc(U.Grid());
    Hloc = Zero();
    for (int mu = 0; mu < Nd; mu++) {
      auto Umu = PeekIndex<LorentzIndex>(U, mu);
      Hloc += trace(Umu * Umu);
    }
    auto Hsum = TensorRemove(sum(Hloc));
    return Hsum.real();
  }

  static inline void Project(Field &U)
  {
      if (isSp2n == true)
      {
          ProjectSp2n(U);
      } else
      {
          ProjectSUn(U);
      }
  }

<<<<<<< HEAD


  static inline void HotConfiguration(GridParallelRNG &pRNG, Field &U)
  {
      Group::HotConfiguration(pRNG, U);
=======
  static inline void HotConfiguration(GridParallelRNG &pRNG, Field &U)
  {
      SU<Nc>::HotConfiguration(pRNG, U);
>>>>>>> 470d4dcc6db9fa41faceac54908bcd60a08bfd8a
  }

  static inline void TepidConfiguration(GridParallelRNG &pRNG, Field &U) {
    Group::TepidConfiguration(pRNG, U);
  }

<<<<<<< HEAD

=======
>>>>>>> 470d4dcc6db9fa41faceac54908bcd60a08bfd8a
  static inline void ColdConfiguration(GridParallelRNG &pRNG, Field &U)
  {
      if (isSp2n == true)
      {
          const int nSp = Nrepresentation/2;
          Sp<nSp>::ColdConfiguration(pRNG, U);
      } else
      {
<<<<<<< HEAD
          Group::ColdConfiguration(pRNG, U);
      }

  }
=======
          SU<Nc>::ColdConfiguration(pRNG, U);
      }
  }
    
  //sp2n... see sp2n.h
/*
    static inline void generate_sp2n_momenta(Field &P, GridSerialRNG & sRNG, GridParallelRNG &pRNG)
    {
      
      const int nSp = Nrepresentation/2;
      LinkField Pmu(P.Grid());
      Pmu = Zero();
      for (int mu = 0; mu < Nd; mu++) {
        
        Sp<nSp>::GaussianFundamentalLieAlgebraMatrix(pRNG, Pmu);
        RealD scale = ::sqrt(HMC_MOMENTUM_DENOMINATOR) ;
        Pmu = Pmu*scale;
        PokeIndex<LorentzIndex>(P, Pmu, mu);
      }
    }
    
    static inline void update_sp2n_field(Field& P, Field& U, double ep){
 
      autoView(U_v,U,AcceleratorWrite);
      autoView(P_v,P,AcceleratorRead);
      accelerator_for(ss, P.Grid()->oSites(),1,{
        for (int mu = 0; mu < Nd; mu++) {
          U_v[ss](mu) = ProjectOnSpGroup(Exponentiate(P_v[ss](mu), ep, Nexp) * U_v[ss](mu));
        }
      });
    }
    
    static inline void SpProject(Field &U) {
      ProjectSp2n(U);
    }


    static inline void ColdSpConfiguration(GridParallelRNG &pRNG, Field &U) {
      const int nSp = Nrepresentation/2;
      Sp<nSp>::ColdConfiguration(pRNG, U);
    }
*/
>>>>>>> 470d4dcc6db9fa41faceac54908bcd60a08bfd8a

};


typedef GaugeImplTypes<vComplex, Nc> GimplTypesR;
typedef GaugeImplTypes<vComplexF, Nc> GimplTypesF;
typedef GaugeImplTypes<vComplexD, Nc> GimplTypesD;

typedef GaugeImplTypes<vComplex, Nc, 12, true> SymplGimplTypesR;
typedef GaugeImplTypes<vComplexF, Nc, 12, true> SymplGimplTypesF;
typedef GaugeImplTypes<vComplexD, Nc, 12, true> SymplGimplTypesD;

<<<<<<< HEAD
typedef GaugeImplTypes<vComplex, Group::AdjointDimension> GimplAdjointTypesR;
typedef GaugeImplTypes<vComplexF, Group::AdjointDimension> GimplAdjointTypesF;
typedef GaugeImplTypes<vComplexD, Group::AdjointDimension> GimplAdjointTypesD;


=======
typedef GaugeImplTypes<vComplex, SU<Nc>::AdjointDimension> GimplAdjointTypesR;
typedef GaugeImplTypes<vComplexF, SU<Nc>::AdjointDimension> GimplAdjointTypesF;
typedef GaugeImplTypes<vComplexD, SU<Nc>::AdjointDimension> GimplAdjointTypesD;
>>>>>>> 470d4dcc6db9fa41faceac54908bcd60a08bfd8a



NAMESPACE_END(Grid);

#endif // GRID_GAUGE_IMPL_TYPES_H
