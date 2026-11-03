#ifndef COARSEWRAPPER_H
#define	COARSEWRAPPER_H

using namespace dolfin;

class CoarseWrapper {
public:
    CoarseWrapper(Mesh &fineMesh, MeshFunction<std::size_t>& fineBounds, MeshFunction<std::size_t>& fineSubdoms, MeshFunction<std::size_t>& facetIntMarkers){
        _fineMesh    = &fineMesh;
        _fineBounds  = &fineBounds;
        _fineSubdoms = &fineSubdoms;
        _fineFacetsMarkers = &facetIntMarkers;
        _fineMesh->init();
        info(*_fineMesh);
    }
    
    int getDim(){
        return _fineMesh->geometry().dim();
    }
  
    Mesh& getFineMesh(){
        return *_fineMesh;
    }
    
    MeshFunction<size_t>& getFineBounds(){
        return *_fineBounds;
    }
    
    MeshFunction<size_t>& getFineSubdoms(){
        return *_fineSubdoms;
    }
    
    MeshFunction<std::size_t>& getInteriorFacetMarkers(){
        return *_fineFacetsMarkers;
    }
    
    int getNumberOfLocalDomains(){
        return _Nw;
    }
    
    
    void findNumberOfLocalDomains(MeshFunction<std::size_t>& cellMarkers){
        std::set<int> markers;
        for(int i = 0; i < cellMarkers.size(); i++){
            int mm = cellMarkers[i];
            markers.insert(mm);
        }
        _Nw = markers.size();
        info("Number of Local Domains %d", _Nw);
    }
    
    void setNumberOfLocalDomains(int Nw){
        _Nw = Nw;
        info("Number of Local Domains %d", _Nw);
    }
    
protected:
    Mesh* _fineMesh;
    MeshFunction<size_t>* _fineSubdoms;
    MeshFunction<size_t>* _fineBounds;
    MeshFunction<size_t>* _fineFacetsMarkers; // for ufl - penalty term
    
    int _Nw;
    
};

#endif	/* COARSEWRAPPER_H */
