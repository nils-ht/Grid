/*
 *  Policy classes for the HMC
 *  Authors: Guido Cossu, David Preti
 */

#ifndef SUN2INDEX_H_H
#define SUN2INDEX_H_H

NAMESPACE_BEGIN(Grid);

/*
 * This is an helper class for the HMC
 * Should contain only the data for the two index representations
 * and the facility to convert from the fundamental -> two index
 * The templated parameter TwoIndexSymmetry choses between the 
 * symmetric and antisymmetric representations
 * 
 * There is an 
 * enum TwoIndexSymmetry { Symmetric = 1, AntiSymmetric = -1 };
 * in the SUnTwoIndex.h file
 */

template <int ncolour, TwoIndexSymmetry S, class group_name = GroupName::SU>
class TwoIndexRep {
public:
  // typdef to be used by the Representations class in HMC to get the
  // types for the higher representation fields
  typedef typename GaugeGroupTwoIndex<ncolour, S, group_name>::LatticeTwoIndexMatrix LatticeMatrix;
  typedef typename GaugeGroupTwoIndex<ncolour, S, group_name>::LatticeTwoIndexField LatticeField;
  static const int Dimension = GaugeGroupTwoIndex<ncolour,S,group_name>::Dimension;
  static const bool isFundamental = false;

  LatticeField U;

  explicit TwoIndexRep(GridBase *grid) : U(grid) {}

  void update_representation(const LatticeGaugeField &Uin) {
    std::cout << GridLogDebug << "Updating TwoIndex representation\n";
    // Uin is in the fundamental representation
    // get the U in TwoIndexRep
    // (U)_{(ij)(lk)} = tr [ adj(e^(ij)) U e^(lk) transpose(U) ]
    conformable(U, Uin);
    U = Zero();
    LatticeColourMatrix tmp(Uin.Grid());

    std::vector<typename GaugeGroup<ncolour,group_name>::Matrix> eij(Dimension);

    for (int a = 0; a < Dimension; a++)
      GaugeGroupTwoIndex<ncolour, S, group_name>::base(a, eij[a]);

    for (int mu = 0; mu < Nd; mu++) {
      auto Uin_mu = peekLorentz(Uin, mu);
      auto U_mu = peekLorentz(U, mu);
      for (int a = 0; a < Dimension; a++) {
        tmp = transpose(Uin_mu) * adj(eij[a]) * Uin_mu;
        for (int b = 0; b < Dimension; b++)
          pokeColour(U_mu, trace(tmp * eij[b]), a, b);
      }
      pokeLorentz(U, U_mu, mu);
    }
  }

  LatticeGaugeField RtoFundamentalProject(const LatticeField &in,
                                          Real scale = 1.0) const {
    LatticeGaugeField out(in.Grid());
    out = Zero();

    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrix out_mu(in.Grid());  // fundamental representation
      LatticeMatrix in_mu = peekLorentz(in, mu);

      out_mu = Zero();

      typename GaugeGroup<ncolour, group_name>::LatticeAlgebraVector h(in.Grid());
      projectOnAlgebra(h, in_mu, double(Nc + 2 * S));  // factor T(r)/T(fund)
      FundamentalLieAlgebraMatrix(h, out_mu);          // apply scale only once
      pokeLorentz(out, out_mu, mu);
    }
    return out;
  }

private:
  void projectOnAlgebra(typename GaugeGroup<ncolour, group_name>::LatticeAlgebraVector &h_out,
                        const LatticeMatrix &in, Real scale = 1.0) const {
    GaugeGroupTwoIndex<ncolour, S,group_name>::projectOnAlgebra(h_out, in, scale);
  }

  void FundamentalLieAlgebraMatrix(
				   typename GaugeGroup<ncolour, group_name>::LatticeAlgebraVector &h,
				   typename GaugeGroup<ncolour, group_name>::LatticeMatrix &out, Real scale = 1.0) const {
    GaugeGroup<ncolour,group_name>::FundamentalLieAlgebraMatrix(h, out, scale);
  }
};

typedef TwoIndexRep<Nc, Symmetric, GroupName::SU> TwoIndexSymmetricRepresentation;
typedef TwoIndexRep<Nc, AntiSymmetric, GroupName::SU> TwoIndexAntiSymmetricRepresentation;

typedef TwoIndexRep<Nc, Symmetric, GroupName::Sp> SpTwoIndexSymmetricRepresentation;
typedef TwoIndexRep<Nc, AntiSymmetric, GroupName::Sp> SpTwoIndexAntiSymmetricRepresentation;

NAMESPACE_END(Grid);

#endif
