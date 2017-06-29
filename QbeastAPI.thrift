namespace py hecuba.qthrift
namespace java es.bsc.qbeast.thrift.api
namespace cpp qview.thrift

struct FilteringArea{
 1:list<double> fromPoint,
 2:list<double> toPoint
}


struct entryPoint{
    1:string blockid,
    2:string hostname,
    3:i32 port
}


enum BasicTypes{
    BIGINT = 0,
    BLOB = 1,
    BOOLEAN = 2,
    DOUBLE = 3,
    FLOAT = 4,
    INET = 5,
    INT = 6,
    LIST = 7,
    MAP = 8,
    SET = 9,
    TEXT = 10,
    TIMESTAMP = 11,
    TIMEUUID = 12,
    UUID = 13,
    DATE = 14,
    TIME = 15,
    DECIMAL=16

}
struct ColumnMeta{
  1:string columnName,
  2:BasicTypes type
}

struct Result{
   1:bool hasMore,
   2:i32 count,
   3:map<byte,ColumnMeta> metadata,
   4:list<map<byte,binary>> data
}

service QbeastMaster{

   string initQuery(
        1:list<string> selects
        2:string keyspace,
        3:string table,
        4:FilteringArea area,
        5:double precision,
        6:i64 maxResults,
        7:list<string> blockIDs
      )
}

exception BlockNotFound{
    1:string message
}
service QbeastWorker{

      /**
       *   3 cases
       *    no data or data< returnAtLeast:
       *      blocks
       *    else
       *     return data
       *
       *
       **/
     Result get(1:string blockID,2:i32 returnAtLeast,3:i32 readMax) throws (1:BlockNotFound blockNotFound)
}
