/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/qcd/action/fermion/WilsonFermion5D.h

    Copyright (C) 2015

Author: Peter Boyle <pabobyle@ph.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
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
#ifndef  GRID_QCD_WILSON_FERMION_5D_H
#define  GRID_QCD_WILSON_FERMION_5D_H

#include <Grid/perfmon/Stat.h>

NAMESPACE_BEGIN(Grid);

////////////////////////////////////////////////////////////////////////////////
// This is the 4d red black case appropriate to support
//
// parity = (x+y+z+t)|2;
// generalised five dim fermions like mobius, zolotarev etc..	
//
// i.e. even even contains fifth dim hopping term.
//
// [DIFFERS from original CPS red black implementation parity = (x+y+z+t+s)|2 ]
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This is the 4d red black case appropriate to support
//
// parity = (x+y+z+t)|2;
// generalised five dim fermions like mobius, zolotarev etc..	
//
// i.e. even even contains fifth dim hopping term.
//
// [DIFFERS from original CPS red black implementation parity = (x+y+z+t+s)|2 ]
////////////////////////////////////////////////////////////////////////////////

class WilsonFermion5DStatic { 
public:
  // S-direction is INNERMOST and takes no part in the parity.
  static const std::vector<int> directions;
  static const std::vector<int> displacements;
  static constexpr int npoint = 8;
};

template<class Impl>
class WilsonFermion5D : public WilsonKernels<Impl>, public WilsonFermion5DStatic
{
public:
  INHERIT_IMPL_TYPES(Impl);
  typedef WilsonKernels<Impl> Kernels;
  PmuStat stat;

  FermionField _tmp;
  FermionField &tmp(void) { return _tmp; }

  int Dirichlet;
  Coordinate Block; 

  ///////////////////////////////////////////////////////////////
  // Implement the abstract base
  ///////////////////////////////////////////////////////////////
  GridBase *GaugeGrid(void)              { return _FourDimGrid ;}
  GridBase *GaugeRedBlackGrid(void)      { return _FourDimRedBlackGrid ;}
  GridBase *FermionGrid(void)            { return _FiveDimGrid;}
  GridBase *FermionRedBlackGrid(void)    { return _FiveDimRedBlackGrid;}

  // full checkerboard operations; leave unimplemented as abstract for now
  virtual void   M    (const FermionField &in, FermionField &out){assert(0);};
  virtual void   Mdag (const FermionField &in, FermionField &out){assert(0);};

  // half checkerboard operations; leave unimplemented as abstract for now
  virtual void   Meooe       (const FermionField &in, FermionField &out){assert(0);};
  virtual void   Mooee       (const FermionField &in, FermionField &out){assert(0);};
  virtual void   MooeeInv    (const FermionField &in, FermionField &out){assert(0);};

  virtual void   MeooeDag    (const FermionField &in, FermionField &out){assert(0);};
  virtual void   MooeeDag    (const FermionField &in, FermionField &out){assert(0);};
  virtual void   MooeeInvDag (const FermionField &in, FermionField &out){assert(0);};
  virtual void   Mdir   (const FermionField &in, FermionField &out,int dir,int disp){assert(0);};   // case by case Wilson, Clover, Cayley, ContFrac, PartFrac
  virtual void   MdirAll(const FermionField &in, std::vector<FermionField> &out){assert(0);};   // case by case Wilson, Clover, Cayley, ContFrac, PartFrac

  // These can be overridden by fancy 5d chiral action
  virtual void DhopDeriv  (GaugeField &mat,const FermionField &U,const FermionField &V,int dag);
  virtual void DhopDerivEO(GaugeField &mat,const FermionField &U,const FermionField &V,int dag);
  virtual void DhopDerivOE(GaugeField &mat,const FermionField &U,const FermionField &V,int dag);

  void MomentumSpacePropagatorHt_5d(FermionField &out,const FermionField &in,RealD mass,std::vector<double> twist) ;
  void MomentumSpacePropagatorHt(FermionField &out,const FermionField &in,RealD mass,std::vector<double> twist) ;
  void MomentumSpacePropagatorHw(FermionField &out,const FermionField &in,RealD mass,std::vector<double> twist) ;
  void MomentumSpacePropagatorHwQ(FermionField &out,const FermionField &in,RealD mass,std::vector<double> twist,
				  std::vector<double> qmu) ;

  // Implement hopping term non-hermitian hopping term; half cb or both
  // Implement s-diagonal DW
  void DW    (const FermionField &in, FermionField &out,int dag);
  void Dhop  (const FermionField &in, FermionField &out,int dag);
  void DhopOE(const FermionField &in, FermionField &out,int dag);
  void DhopEO(const FermionField &in, FermionField &out,int dag);

  void DhopComms  (const FermionField &in, FermionField &out);
  void DhopCalc   (const FermionField &in, FermionField &out,uint64_t *ids);
  
  // add a DhopComm
  // -- suboptimal interface will presently trigger multiple comms.
  void DhopDir(const FermionField &in, FermionField &out,int dir,int disp);
  void DhopDirAll(const FermionField &in,std::vector<FermionField> &out);
  void DhopDirComms(const FermionField &in);
  void DhopDirCalc(const FermionField &in, FermionField &out,int point);
    
  ///////////////////////////////////////////////////////////////
  // New methods added 
  ///////////////////////////////////////////////////////////////
  void DerivInternal(StencilImpl & st,
		     DoubledGaugeField & U,
		     GaugeField &mat,
		     const FermionField &A,
		     const FermionField &B,
		     int dag);
    
  void DhopInternal(StencilImpl & st,
		    DoubledGaugeField &U,
		    const FermionField &in, 
		    FermionField &out,
		    int dag);

  void DhopInternalOverlappedComms(StencilImpl & st,
				   DoubledGaugeField &U,
				   const FermionField &in, 
				   FermionField &out,
				   int dag);

  void DhopInternalSerialComms(StencilImpl & st,
			       DoubledGaugeField &U,
			       const FermionField &in, 
			       FermionField &out,
			       int dag);
    
  // Constructors
  WilsonFermion5D(GaugeField &_Umu,
		  GridCartesian         &FiveDimGrid,
		  GridRedBlackCartesian &FiveDimRedBlackGrid,
		  GridCartesian         &FourDimGrid,
		  GridRedBlackCartesian &FourDimRedBlackGrid,
		  double _M5,const ImplParams &p= ImplParams());

  virtual void DirichletBlock(const Coordinate & block)
  {
  }
  // Constructors
  /*
    WilsonFermion5D(int simd, 
    GaugeField &_Umu,
    GridCartesian         &FiveDimGrid,
    GridRedBlackCartesian &FiveDimRedBlackGrid,
    GridCartesian         &FourDimGrid,
    double _M5,const ImplParams &p= ImplParams());
  */
    
  // DoubleStore
  void ImportGauge(const GaugeField &_Umu);
    
  ///////////////////////////////////////////////////////////////
  // Data members require to support the functionality
  ///////////////////////////////////////////////////////////////
public:
    
  // Add these to the support from Wilson
  GridBase *_FourDimGrid;
  GridBase *_FourDimRedBlackGrid;
  GridBase *_FiveDimGrid;
  GridBase *_FiveDimRedBlackGrid;
    
  double                        M5;
  int Ls;
    
  //Defines the stencils for even and odd
  StencilImpl Stencil; 
  StencilImpl StencilEven; 
  StencilImpl StencilOdd; 
    
  // Copy of the gauge field , with even and odd subsets
  DoubledGaugeField Umu;
  DoubledGaugeField UmuEven;
  DoubledGaugeField UmuOdd;
    
  // Comms buffer
  //  std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  comm_buf;


};

NAMESPACE_END(Grid);

#endif
