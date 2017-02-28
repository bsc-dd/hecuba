package storage;

import com.datastax.driver.core.*;

import java.util.*;
import java.util.function.Function;

import static java.util.stream.Collectors.*;


public class StorageItf {

    private static Cluster cluster = null;
    private static Session session = null;
    private static StorageItf client;
    private static String[] nodeIP;
    private static Integer nodePort;
    private static String configFile;

    private final static String version;

    static {
        String v=System.getenv("HECUBA_VERSION");
        if(v==null){
            version="1.0";
        }else{
            version=v;
        }
    }


    /**
     * This function returns a order list of the hosts that have the higher number of local tokens.
     *
     * @param objectID the block identifier
     * @return
     * @throws storage.StorageException
     */
    public static List<String> getLocations(String objectID) throws storage.StorageException {
        if(version.equals("1.0")){
            return getLocationsV1(objectID);
        }else if(version.matches("2\\.[0-9]+")){
            return getLocationsV2(objectID);
        } else {
            throw new StorageException("UNKNOWN HECUBA VERSION: "+version);
        }

    }
    public static List<String> getLocationsV2(String objectID) throws storage.StorageException {
        objectID = objectID.replace(" ", "");
        String[] need = objectID.split("_");
        int needLen = need.length;

        if (needLen == 2) { //storageObj
            List<String> resultSet = Collections.<String>emptyList();
            return resultSet;
        } else {
            //if (needLen == 1) block
            checkCassandra();

            Metadata metadata = cluster.getMetadata();
            String name = session.execute("SELECT name FROM hecuba.istorage WHERE storage_id = ?", objectID)
                    .one().getString("name");
            int pposition=name.indexOf('.');
            if(pposition==-1){
                throw new StorageException("I cannot detect the keyspace name from "+name);
            }
            final String nodeKp=name.substring(0,pposition);
            System.out.println(nodeKp);

            Set<Map.Entry<Host, Long>> hostsTkns = session.execute("SELECT tokens FROM hecuba.istorage WHERE storage_id = ?", objectID)
                    .one().getList("tokens", TupleValue.class).stream()
                    .map(tok -> metadata.newToken(tok.getVarint(0).toString()))
                    .flatMap(token ->
                            metadata.getReplicas(Metadata.quote(nodeKp), metadata.newTokenRange(token, token)).stream())
                    .collect(groupingBy(Function.identity(), counting())).entrySet();

            ArrayList<Map.Entry<Host, Long>> result = new ArrayList<>(hostsTkns);
            Collections.sort(result, Comparator.comparing(o -> (o.getValue())));
            List<String> toReturn;
            toReturn = result.stream().map(a -> a.getKey().getAddress().toString().replace("/", "")).collect(toList());
            return toReturn;
        }
    }

    public static List<String> getLocationsV1(String objectID) throws storage.StorageException {
        objectID = objectID.replace(" ", "");
        String[] need = objectID.split("_");
        int needLen = need.length;

        if (needLen == 2) { //storageObj
            List<String> resultSet = Collections.<String>emptyList();
            return resultSet;
        } else {
            //if (needLen == 1) block
            checkCassandra();

            Row row = session.execute("SELECT entry_point FROM hecuba.blocks WHERE blockid = ?", objectID).one();
            if (row == null) {

                throw new storage.StorageException("Block " + objectID + " not found");
            }

            String hostEntryPoint = row.getString("entry_point");
            if (hostEntryPoint != null) {
                //The host point is already defined.
                List<String> result = new ArrayList<>();
                for(int i=0;i<3;i++)
                    result.add(hostEntryPoint);
                return result;
            }


            Metadata metadata = cluster.getMetadata();
            String nodeKp = session.execute("SELECT ksp FROM hecuba.blocks WHERE blockid = ?", objectID).one().getString("ksp");
            System.out.println(nodeKp);

            Set<Map.Entry<Host, Long>> hostsTkns = session.execute("SELECT tkns FROM hecuba.blocks WHERE blockid = ?", objectID)
                    .one().getList("tkns", Long.class).stream().map(tok -> metadata.newToken(tok.toString()))
                    .flatMap(token ->
                            metadata.getReplicas(Metadata.quote(nodeKp), metadata.newTokenRange(token, token)).stream())
                    .collect(groupingBy(Function.identity(), counting())).entrySet();

            ArrayList<Map.Entry<Host, Long>> result = new ArrayList<>(hostsTkns);
            Collections.sort(result, Comparator.comparing(o -> (o.getValue())));
            List<String> toReturn;                                                 
            toReturn = result.stream().map(a -> a.getKey().getAddress().toString().replace("/", "")).collect(toList());
            return toReturn;                                                                                           
        }
    }

    private static void checkCassandra() {
        if (cluster == null) {
            System.out.println(nodeIP);
            cluster = new Cluster.Builder()
                    .addContactPoints(nodeIP)
                    .withPort(nodePort)
                    .build();
            session = cluster.connect();
        }

    }

    private static void closeCassandra() {
        if (cluster != null) {
            session.close();
            session = null;
            cluster.close();
            cluster = null;
        }

    }


    public static void newReplica(String objectID, String node) throws StorageException {
    }

    public static String newVersion(String objectID, String node) throws StorageException {
        return "";
    }

    public static void consolidateVersion(String objectID) throws StorageException {
    }

    public static void delete(String objectID) throws StorageException {
    }

    public static void finish() throws StorageException {
        //closeCassandra();

    }

    public static java.lang.Object getByID(String objectID) throws StorageException {
        return null;
    }

    public static void init(String configFile) throws storage.StorageException {
        nodeIP = System.getenv("CONTACT_NAMES").split(",");
        nodePort = Integer.parseInt(System.getenv("NODE_PORT"));
    }

    public static void main(String[] args) {

        configFile = System.getProperty("user.home") + "/hecuba2/hecuba/__init__.py";
        System.out.println("configFile:" + configFile);
        client = new StorageItf();

        try {
            client.init(configFile);
        } catch (StorageException e) {
            e.printStackTrace();
        }
        try {
            System.out.println(client.getLocations("5dad0db4-c08a-11e6-9385-001517e6a1fc"));
            System.out.println(client.getLocations("Words_1"));
        } catch (StorageException e) {
            e.printStackTrace();
        }
        System.out.println("Application ended");
    }
}
