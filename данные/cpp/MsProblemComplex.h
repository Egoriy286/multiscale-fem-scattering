#ifndef GMSFEMPROBLEM_H
#define GMSFEMPROBLEM_H

#include <math.h>
#include <iostream>
#include <fstream>
#include <dolfin.h>
#include "MsBlock.h"
#include "CoarseWrapper.h"

using namespace dolfin;

class MsProblemComplex{
public:

    MsProblemComplex(CoarseWrapper& coarseWrapper){
        _meshWrapper = &coarseWrapper;
    };
    
    void addDirichletBC(GenericFunction& g, size_t sub_domain, std::string type = "topological") {
         _dirichletBC.push_back(new DirichletBC(*_V, g, _meshWrapper->getFineBounds(), sub_domain, type));
    };
    
    /*
     * Fine-scale solver 
     */
    void generateSystem(){
        info("-------------generateSystem-------------");
        // --- lhs --- 
        assemble(*_ARe, *_are);
        assemble(*_AIm, *_aim);
        for (int i = 0; i < _dirichletBC.size(); i++) {
            (*_dirichletBC[i]).apply(*_ARe);
        }
        modifyA();
//         --- rhs --- 
        assemble(*_bRe, *_Lre); 
        assemble(*_bIm, *_Lim); 
        for (int i = 0; i < _dirichletBC.size(); i++) {
            (*_dirichletBC[i]).apply(*_bRe);
        }
        modifyb();
        saveMatrixBin(*_ARe, "binFiles/ARe");
        saveMatrixBin(*_AIm, "binFiles/AIm");
        saveVectorBin(*_bRe, "binFiles/bRe.dat");
        saveVectorBin(*_bIm, "binFiles/bIm.dat");
    }
    
    void saveToPvd(std::string inname, std::string outname){
        int dim =  _u->vector().get()->size();
        info("moved %d", dim);
        
        PetscScalar avec[dim];
        int fd;
        PetscViewer viewerRhs1;
        PetscViewerBinaryOpen(PETSC_COMM_SELF, inname.data(), FILE_MODE_READ, &viewerRhs1);
        PetscViewerBinaryGetDescriptor(viewerRhs1, &fd);
        PetscBinaryRead(fd, avec, dim, PETSC_DOUBLE);//PETSC_SCALAR);
    //    VecLoad(b2, viewerRhs1);
        printf("Load vector size %d \n", dim);
        
        for(int i = 0; i < dim; i++){
            double val = avec[i];
            _u->vector().get()->setitem(i, val);
//            info("%d %g", i, val);
        }
        _u->vector().get()->apply("insert");
        
        saveOut(outname);

        PetscViewerDestroy(&viewerRhs1);
        info("moved");
    };
    
    /*  
     * Coarse-scale solver
     */
    Mat PR, PRT;
    int eCount; //number of basis
    std::vector<MsBlock*> msblocks;
    virtual void loadMsBasis(std::string bDir, std::string bDirEigen){
        msblocks.clear();
        int subDomCount = _meshWrapper->getNumberOfLocalDomains();
        for(int i = 0; i < subDomCount; i++){
            MsBlock *msblock = new MsBlock(i, bDir, eCount);
            msblock->loadDOF();
            msblock->loadMSBasis(bDirEigen);
            msblock->loadMSBasisDG(bDirEigen);// for DG
            msblocks.push_back(msblock);
        }
        info("load ms basis");
    }

     virtual void loadMsBasis3(std::string bDir, std::string bDirEigen){
        msblocks.clear();
        int subDomCount = _meshWrapper->getNumberOfLocalDomains();
        for(int i = 0; i < subDomCount; i++){
            MsBlock *msblock = new MsBlock(i, bDir, eCount);
            msblock->loadDOF();
            msblock->loadMSBasisDG(bDirEigen);
            msblocks.push_back(msblock);
        }
        info("load ms basis");
    }
    
    int eCount2;
    std::vector<MsBlock*> msblocks2;
    virtual void loadMsBasis2(std::string bDir, std::string bDirEigen, std::string bDirEigen2){
        info("-------------loadMsBasis2-------------");
        msblocks.clear();
        int subDomCount = _meshWrapper->getNumberOfLocalDomains();
        for(int i = 0; i < subDomCount; i++){
            MsBlock *msblock = new MsBlock(i, bDir, eCount);
            msblock->loadDOF();
            msblock->loadMSBasisDG(bDirEigen);// for DG
            msblocks.push_back(msblock);
        }
        info("load ms basis b");

        msblocks2.clear();
        for(int i = 0; i < subDomCount; i++){
            MsBlock *msblock = new MsBlock(i, bDir, eCount2);
            msblock->loadDOF();
            msblock->loadMSBasisDG(bDirEigen2);
            msblocks2.push_back(msblock);
        }
        info("load ms basis i");
    }
    
    virtual void loadMsBasisByLambda(std::string bDir, std::string bDirEigen){
        msblocks.clear();
        int subDomCount = _meshWrapper->getNumberOfLocalDomains();
        for(int i = 0; i < subDomCount; i++){
            int bcount = getNumberOfBasisByLambda(bDir, i);
            bcount = (bcount < 1)?1:bcount;
            info("-- For %d take %d basis", i, bcount);
            MsBlock *msblock = new MsBlock(i, bDir, bcount);
            msblock->loadDOF();
            msblock->loadMSBasis(bDirEigen);
            msblocks.push_back(msblock);
        }
        info("load ms bases by lmd");
    }


    int getNumberOfBasisByLambda(std::string basisDir, int i){
        double lmbLimit = 1.e-1;//1.0e-1;
        std::ifstream ifile(basisDir + "lmd" + std::to_string(i));
        // std::string dd = basisDir + "lmd" + std::to_string(i);
        // info("%s", dd.data());
        int bCount = 0;
        double lmd, lmdPrev;
        ifile >> lmd;
        for(int l = 1; l < eCount; l++){
            lmdPrev = lmd;
            ifile >> lmd;
            // info("lmd %g", lmd);
            if(lmd > lmbLimit){
//            if(lmd > lmbLimit && l > 1){// l-1
                bCount = l+1;//l+1;//l-1;//l
                break;
            }
        }
        ifile.close();
        return bCount;
    }
    
    virtual void generateR(){
        info("-------------generateR-------------");
        int rowCount = 0;
        int maxNZ = 0;
        for(int bi = 0; bi < msblocks.size(); bi++){
            MsBlock* block = msblocks.at(bi);
            rowCount += block->getCoarseSize();
            //info("rowCount = %d",rowCount);
            maxNZ = (maxNZ < block->getFineSize())?block->getFineSize():maxNZ;
        }
        int colCount = _u->vector().get()->size();
        MatCreateSeqAIJ(PETSC_COMM_SELF, rowCount, colCount, maxNZ, 0, &PR);

        int counter = 0;
       
//        int Nrow = eCount;
        for(int bi = 0; bi < msblocks.size(); bi++){
            MsBlock* block = msblocks.at(bi);
            int Nrow = block->getCoarseSize();
            for(int i = 0; i < Nrow; i++){
                int Ncol = block->getFineSize();
                for(int j = 0; j < Ncol; j++){
                    int gi = block->getGlobalByLocal(j);
                    double val = block->getRItem(i, j); 
                    MatSetValues(PR, 1, &counter, 1, &gi ,&val, INSERT_VALUES);
                }
                counter++;
            }
        }
        MatAssemblyBegin(PR, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(PR, MAT_FINAL_ASSEMBLY);
        
        MatTranspose(PR, MAT_INITIAL_MATRIX, &PRT);   
        info("generate global R[%d, %d]", rowCount, colCount);
    }
    
    // DG - interior and boundary
    virtual void generateR2(){
        info("-------------generateR2-------------");
        std::ofstream fout1("vals1.txt", std::ios_base::out);
        std::ofstream fout2("vals2.txt", std::ios_base::out);
        int maxNZ = 0;
        for(int bi = 0; bi < msblocks.size(); bi++){
            MsBlock* block = msblocks.at(bi);
            maxNZ = (maxNZ < block->getFineSize())?block->getFineSize():maxNZ;
        }
        int rowCount = msblocks.size()*eCount + msblocks2.size()*eCount2;
        int colCount = _u->vector().get()->size();
        MatCreateSeqAIJ(PETSC_COMM_SELF, rowCount, colCount, maxNZ, 0, &PR);
        info("global R %d, %d and nnz = %d", rowCount, colCount, maxNZ);

        int counter = 0;
        // int Nrow = eCount;
        // 1            boundary
        for(int bi = 0; bi < msblocks.size(); bi++){
            MsBlock* block = msblocks.at(bi);
            int Nrow = block->getCoarseSize();
            //info("NORW = %i", Nrow);
            for(int i = 0; i < Nrow; i++){
                int Ncol = block->getFineSize();
                //info("NCOL = %d", Ncol);
                for(int j = 0; j < Ncol; j++){
                    int gi = block->getGlobalByLocal(j);
                    double val = block->getRItem(i, j);
                    //info("GI = %d,   VAL = %f", gi, val);
                                        
                    MatSetValues(PR, 1, &counter, 1, &gi ,&val, INSERT_VALUES);
                    fout1<<"counter = "<<counter<<"  "<<"gi = "<<gi<<"   "<<"val = "<<val<<"\n";
                }
                counter++;
                //info("counter = %d", counter);
                fout1.close(); 
            }
        }
        // 2            interior
        for(int bi = 0; bi < msblocks2.size(); bi++){
            MsBlock* block = msblocks2.at(bi);
            int Nrow = block->getCoarseSize();
            for(int i = 0; i < Nrow; i++){
                int Ncol = block->getFineSize();
                for(int j = 0; j < Ncol; j++){
                    int gi = block->getGlobalByLocal(j);
                    double val = block->getRItem(i, j); 
                    
                    MatSetValues(PR, 1, &counter, 1, &gi ,&val, INSERT_VALUES);
                    fout2<<"counter = "<<counter<<"  "<<"gi = "<<gi<<"   "<<"val = "<<val<<"\n";
                }
                counter++;
                //info("counter = %d", counter);
                fout2.close(); 
                //info("*********** NROW = %d, NCOL = %d", Nrow, Ncol);
            }
        }
        
        MatAssemblyBegin(PR, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(PR, MAT_FINAL_ASSEMBLY);
        info("1111111111111");
        MatTranspose(PR, MAT_INITIAL_MATRIX, &PRT); 
        info("generate global R[%d, %d]", rowCount, colCount);
        
    }
    
    void generateSystemCoarse(){
        info("-------------generateSystemCoarse-------------");
        PetscInt dim, dim2;
        MatGetSize(PR, &dim, &dim2);
        info("generate %d", dim);
        
        Mat PRA1, PRA1RT;
        Mat PA1 = _ARe->mat();
        MatMatMult(PR, PA1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PRA1);
        MatMatMult(PRA1, PRT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PRA1RT);
        info("generate ARe_c");
        Mat PRA2, PRA2RT;
        Mat PA2 = _AIm->mat();
        MatMatMult(PR, PA2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PRA2);
        MatMatMult(PRA2, PRT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PRA2RT);
        info("generate AIm_c");
        
        Vec PRb1;
        VecCreateSeq(PETSC_COMM_SELF, dim, &PRb1);
        Vec Pb1 = _bRe->vec();
        MatMult(PR, Pb1, PRb1);
        info("generate bRe_c");
        Vec PRb2;
        VecCreateSeq(PETSC_COMM_SELF, dim, &PRb2);
        Vec Pb2 = _bIm->vec();
        MatMult(PR, Pb2, PRb2);
        info("generate bIm_c");

        saveMatrixBin(PRA1RT, "binFiles/ARe");
        saveMatrixBin(PRA2RT, "binFiles/AIm");
        saveVectorBin(PRb1, "binFiles/bRe.dat");
        saveVectorBin(PRb2, "binFiles/bIm.dat");
        info("saved");
    }
    
    void saveToPvdCoarse(std::string inname, std::string outname){
        info("-------------saveToPvdCoarse-------------");
        PetscInt dim, dim2;
        MatGetSize(PR, &dim, &dim2);
        info("moved %d", dim);
        
        PetscScalar avec[dim];
        int fd;
        PetscViewer viewerRhs1;
        PetscViewerBinaryOpen(PETSC_COMM_SELF, inname.data(), FILE_MODE_READ, &viewerRhs1);
        PetscViewerBinaryGetDescriptor(viewerRhs1, &fd);
        PetscBinaryRead(fd, avec, dim, PETSC_DOUBLE);//PETSC_SCALAR);
    //    VecLoad(b2, viewerRhs1);
        printf("Load vector size %d \n", dim);
        
        Vec Puc;
        VecCreateSeq(PETSC_COMM_SELF, dim, &Puc);
        for(int j = 0; j < dim; j++){
            VecSetValues(Puc, 1, &j, &avec[j], INSERT_VALUES);
//            info("%g", avec[j]);
        }
        VecAssemblyBegin(Puc);
        VecAssemblyEnd(Puc);

        // to fine grid
        Vec Pums;
        VecCreateSeq(PETSC_COMM_SELF, dim2, &Pums);
        MatMult(PRT, Puc, Pums);

        double val;        
        for(int i = 0; i < dim2; i++){
            VecGetValues(Pums, 1, &i, &val);
            _u->vector().get()->setitem(i, val);
        //    info("%d %g", i, val);
        }
        _u->vector().get()->apply("insert");
        
        saveOut(outname);

        PetscViewerDestroy(&viewerRhs1);
        info("moved");
    };
    
    virtual void saveOut(std::string outname){
        File file(outname);
        file << u();
    }
    
    FunctionSpace& V() {
        return *_V;
    }
    
    Function& u() {
        return *_u;
    }
    
protected:
    FunctionSpace* _V;
    Function* _u;
    CoarseWrapper* _meshWrapper;
    std::vector<DirichletBC*> _dirichletBC;
    
    Form* _are;
    Form* _aim;
    Form* _Lre;
    Form* _Lim;
    PETScMatrix* _ARe;
    PETScMatrix* _AIm;
    PETScVector* _bRe;
    PETScVector* _bIm;
    
    virtual void modifyA(){
        // DO NOTHING
    }
    
    virtual void modifyb(){
        // DO NOTHING
    }
    
    void saveMatrixBin(PETScMatrix& S, std::string fname){
        Mat PA1 = S.mat();
        saveMatrixBin(PA1, fname);
    }
    
    void saveMatrixBin(Mat& PA1, std::string fname){
        info("-------------saveMatrixBin-------------");
        int fd, fd1, fd2;
        std::string mname1 = fname + "1.dat";
        std::string mname2 = fname + "2.dat";
        std::string mnameo = fname + "o.dat";
        info("START: save %s", fname.data());

        PetscInt dim1, dim2;
        MatGetSize(PA1, &dim1, &dim2);
        info("dim1 = %d", dim1);
        info("dim2 = %d", dim2);
        
        PetscInt ncols;
        const PetscInt *cols;
        const PetscReal *vals;
        PetscInt olen[dim1];
        int maxnz = 0;
        for(int row = 0; row < dim1; row++){ // rows
            MatGetRow(PA1, row, &ncols, &cols, &vals);
            maxnz = (maxnz < ncols)?ncols:maxnz;
            olen[row] = ncols;
        }
        PetscViewer viewer1o;
        PetscViewerBinaryOpen(PETSC_COMM_SELF, mnameo.data(), FILE_MODE_WRITE, &viewer1o);
        PetscViewerBinaryGetDescriptor(viewer1o, &fd);
        PetscBinaryWrite(fd, &olen, dim1, PETSC_INT, PETSC_FALSE);
        PetscViewerDestroy(&viewer1o);
                 
        PetscViewer viewer11, viewer12;
        PetscViewerBinaryOpen(PETSC_COMM_SELF, mname1.data(), FILE_MODE_WRITE, &viewer11);
        PetscViewerBinaryOpen(PETSC_COMM_SELF, mname2.data(), FILE_MODE_WRITE, &viewer12);
        PetscViewerBinaryGetDescriptor(viewer11, &fd1);
        PetscViewerBinaryGetDescriptor(viewer12, &fd2);
        PetscReal vals2[maxnz];
        PetscInt cols2[maxnz];
        for(int row = 0; row < dim1; row++){ // rows
            MatGetRow(PA1, row, &ncols, &cols, &vals);
            for(int j = 0; j < ncols; j++){
                vals2[j] = vals[j];
                cols2[j] = cols[j];
            }
            PetscBinaryWrite(fd1, vals2, ncols, PETSC_DOUBLE, PETSC_FALSE);//PETSC_SCALAR, PETSC_FALSE);
            PetscBinaryWrite(fd2, cols2, ncols, PETSC_INT, PETSC_FALSE);//PETSC_SCALAR, PETSC_FALSE);
//            info("--- ncols %d ---", ncols);
//            for(int j = 0; j < ncols; j++){
////                info("c %d, v %g", cols[j], vals[j]);
//                if(std::abs(vals[j]) > 1.0e-10)
//                    info("%d with nz %d: c %d, v %g", row, ncols, cols[j], vals[j]);
//            }
        }
//      MatView(PA1, viewer1);
        info("END: save %s [%d, %d] with nz %d", fname.data(), dim1, dim2, maxnz);
        PetscViewerDestroy(&viewer11);
        PetscViewerDestroy(&viewer12);
    }
    
    void saveVectorBin(PETScVector& rhs, std::string mname){
        Vec Pb1 = rhs.vec();
        saveVectorBin(Pb1, mname);
    }
    
    void saveVectorBin(Vec& Pb1, std::string mname){
        int dim;
        VecGetSize(Pb1, &dim);
        PetscScalar *array;
        VecGetArray(Pb1, &array);

        int fd;
        PetscViewer viewerRhs2;
        PetscViewerBinaryOpen(PETSC_COMM_SELF, mname.data(), FILE_MODE_WRITE, &viewerRhs2);
        PetscViewerBinaryGetDescriptor(viewerRhs2, &fd);
        PetscBinaryWrite(fd, array, dim, PETSC_DOUBLE, PETSC_FALSE);//PETSC_SCALAR, PETSC_FALSE);
//        VecView(Pb1, viewerRhs2);
        info("save %s %d", mname.data(), dim);
        PetscViewerDestroy(&viewerRhs2);
    }
    
};

#endif
