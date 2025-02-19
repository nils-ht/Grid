/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/qcd/action/gauge/PlaqPlusRectangleAction.h

    Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
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

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */
#ifndef QCD_PLAQ_PLUS_RECTANGLE_ACTION_H
#define QCD_PLAQ_PLUS_RECTANGLE_ACTION_H

NAMESPACE_BEGIN(Grid);  
    
////////////////////////////////////////////////////////////////////////
// PlaqPlusRectangleActoin
////////////////////////////////////////////////////////////////////////
template<class Gimpl>
class PlaqPlusRectangleAction : public Action<typename Gimpl::GaugeField> {
public:

  INHERIT_GIMPL_TYPES(Gimpl);

  using Action<GaugeField>::S;
  using Action<GaugeField>::Sinitial;
  using Action<GaugeField>::deriv;
  using Action<GaugeField>::refresh;

private:
  RealD c_plaq;
  RealD c_rect;
  typename WilsonLoops<Gimpl>::StapleAndRectStapleAllWorkspace workspace;
public:
  PlaqPlusRectangleAction(RealD b,RealD c): c_plaq(b),c_rect(c){};

  virtual std::string action_name(){return "PlaqPlusRectangleAction";}
      
  virtual void refresh(const GaugeField &U, GridSerialRNG &sRNG, GridParallelRNG& pRNG) {}; // noop as no pseudoferms
      
  virtual std::string LogParameters(){
    std::stringstream sstream;
    sstream << GridLogMessage << "["<<action_name() <<"] c_plaq: " << c_plaq << std::endl;
    sstream << GridLogMessage << "["<<action_name() <<"] c_rect: " << c_rect << std::endl;
    return sstream.str();
  }


  virtual RealD S(const GaugeField &U) {
    RealD vol = U.Grid()->gSites();

    RealD plaq = WilsonLoops<Gimpl>::avgPlaquette(U);
    RealD rect = WilsonLoops<Gimpl>::avgRectangle(U);

    RealD action=c_plaq*(1.0 -plaq)*(Nd*(Nd-1.0))*vol*0.5
      +c_rect*(1.0 -rect)*(Nd*(Nd-1.0))*vol;

    return action;
  };

  virtual void deriv(const GaugeField &Umu,GaugeField & dSdU) {
    //extend Ta to include Lorentz indexes
    RealD factor_p = c_plaq/RealD(Nc)*0.5;
    RealD factor_r = c_rect/RealD(Nc)*0.5;

    GridBase *grid = Umu.Grid();

    std::vector<GaugeLinkField> U (Nd,grid);
    for(int mu=0;mu<Nd;mu++){
      U[mu] = PeekIndex<LorentzIndex>(Umu,mu);
    }
    std::vector<GaugeLinkField> RectStaple(Nd,grid), Staple(Nd,grid);
    WilsonLoops<Gimpl>::StapleAndRectStapleAll(Staple, RectStaple, U, workspace);

    GaugeLinkField dSdU_mu(grid);
    GaugeLinkField staple(grid);

    for (int mu=0; mu < Nd; mu++){
      dSdU_mu = Ta(U[mu]*Staple[mu])*factor_p;
      dSdU_mu = dSdU_mu + Ta(U[mu]*RectStaple[mu])*factor_r;
	  
      PokeIndex<LorentzIndex>(dSdU, dSdU_mu, mu);
    }

  };

};

// Convenience for common physically defined cases.
//
// RBC c1 parameterisation is not really RBC but don't have good
// reference and we are happy to change name if prior use of this plaq coeff
// parameterisation is made known to us. 
template<class Gimpl>
class RBCGaugeAction : public PlaqPlusRectangleAction<Gimpl> {
public:
  INHERIT_GIMPL_TYPES(Gimpl);
  RBCGaugeAction(RealD beta,RealD c1) : PlaqPlusRectangleAction<Gimpl>(beta*(1.0-8.0*c1), beta*c1) {};
  virtual std::string action_name(){return "RBCGaugeAction";}
};

template<class Gimpl>
class IwasakiGaugeAction : public RBCGaugeAction<Gimpl> {
public:
  INHERIT_GIMPL_TYPES(Gimpl);
  IwasakiGaugeAction(RealD beta) : RBCGaugeAction<Gimpl>(beta,-0.331) {};
  virtual std::string action_name(){return "IwasakiGaugeAction";}
};

template<class Gimpl>
class SymanzikGaugeAction : public RBCGaugeAction<Gimpl> {
public:
  INHERIT_GIMPL_TYPES(Gimpl);
  SymanzikGaugeAction(RealD beta) : RBCGaugeAction<Gimpl>(beta,-1.0/12.0) {};
  virtual std::string action_name(){return "SymanzikGaugeAction";}
};

template<class Gimpl>
class DBW2GaugeAction : public RBCGaugeAction<Gimpl> {
public:
  INHERIT_GIMPL_TYPES(Gimpl);
  DBW2GaugeAction(RealD beta) : RBCGaugeAction<Gimpl>(beta,-1.4067) {};
  virtual std::string action_name(){return "DBW2GaugeAction";}
};

NAMESPACE_END(Grid);

#endif
