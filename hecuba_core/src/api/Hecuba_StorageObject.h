#ifndef __HECUBA_STORAGE_OBJECT_H__
#define  __HECUBA_STORAGE_OBJECT_H__

#include "UUID.h"
#include "ObjSpec.h"
#include "IStorage.h"
#include "SORecursiveMagic.h"


class StorageObject: virtual public IStorage{

    public:

        StorageObject();
        // c++ only calls implicitly the constructor without parameters. To invoke this constructor we need to add to the user class an explicit call to this
        StorageObject(const std::string& name);
        StorageObject(const StorageObject& src);
        ~StorageObject();
        StorageObject& operator = (const StorageObject& src);


        // It generates the python specification for the class during the registration of the object
        void generatePythonSpec();

        void assignTableName(const std::string& id_obj, const std::string& class_name);

        void persist_metadata(uint64_t* c_uuid) ;

        /* setPersistence - Inicializes current instance to conform to uuid object. To be used on an empty instance. */
        void setPersistence (uint64_t *uuid) ;

        void initialize_dataAcces();

        /* Return:
         *  memory reference to datatype (must be freed by user) */
        void getAttr(const std::string&  attr_name, void* valuetoreturn);


        void setAttr(const std::string& attr_name, void* value);

        void setAttr(const std::string& attr_name, IStorage* value);


        void addAttrSpec(const std::string& type, const std::string& name);


        ObjSpec& getObjSpec();

    private:
        ObjSpec generateObjSpec();
        //valuesDesc is used to generate the python definition of the class and to store the attributes description in Cassandra
        std::vector<std::pair<std::string, std::string>> valuesDesc;


};


#endif  // __HECUBA_STORAGE_OBJECT_H__
