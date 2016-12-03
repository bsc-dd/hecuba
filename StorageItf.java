package storage;

import com.datastax.driver.core.*;

import java.util.*;
import java.util.function.Function;
import java.io.*;

import static java.util.stream.Collectors.*;


public class StorageItf {

    public static Cluster cluster = null;
    public static Session session = null;
    public static StorageItf client;
    public static String[] nodeIP;
    public static Integer nodePort;
    public static String configFile;

    /**
    * This function returns a order list of the hosts that have the higher number of local tokens.
    *
    * @param objectID the block identifier
    * @return
    * @throws storage.StorageException
    */
    public static List<String> getLocations(String objectID) throws storage.StorageException {
        objectID = objectID.replace(" ", "");
        String[] need = objectID.split("_");
        int needLen = need.length;
        int storObjLen1 = 1;
        int storObjLen2 = 2;

        if(needLen == storObjLen2){ //storageObj
            List<String> resultSet = Collections.<String>emptyList();
            return resultSet;
        }
        else{ //block
            checkCassandra();
            if(needLen == storObjLen1){
                Row row=session.execute("SELECT entry_point FROM hecuba.blocks WHERE blockid = ?", objectID).one();
                if(row!=null){
                    String hostEntryPoint = row.getString("entry_point");
                    if(hostEntryPoint != null){
                        List<String> result = new ArrayList<>();
                        result.add(hostEntryPoint);
                        return result;
                    }
                }
                Metadata metadata = cluster.getMetadata();
                String nodeKp = session.execute("SELECT ksp FROM hecuba.blocks WHERE blockid = ?", objectID).one().getString("ksp");
                System.out.println(nodeKp);

                Set<Map.Entry<Host, Long>> hostsTkns = session.execute("SELECT tkns FROM hecuba.blocks WHERE blockid = ?", objectID)
                        .one().getList("tkns",Long.class).stream().map(tok -> metadata.newToken(tok.toString()))
                        .flatMap(token ->
                                metadata.getReplicas(Metadata.quote(nodeKp), metadata.newTokenRange(token, token)).stream())
                        .collect(groupingBy(Function.identity(), counting())).entrySet();

                ArrayList<Map.Entry<Host, Long>> result = new ArrayList<>(hostsTkns);
                Collections.sort(result, (o1, o2) -> (o1.getValue()).compareTo(o2.getValue()));
                return result.stream().map(a -> a.getKey().getAddress().toString().replace("/","")).collect(toList());
            }
            else{
                List<String> resultSet    = new ArrayList<String>();
                String nodeKp = need[0];
                List nodeTok = new ArrayList();
                Integer ind = 0;
                for(String val : need){
                    if (ind >= 3){
                        nodeTok.add(val);
                    }
                    ind += 1;
                }

                Metadata metadata = cluster.getMetadata();
                Set<TokenRange> allRanges = metadata.getTokenRanges();
                Integer rangeInd = 0;
                String start_token = "";
                Integer found = 0;
                for (TokenRange tokR : allRanges){
                    for (Object blockRange : nodeTok){
                        if (rangeInd.toString().equals(blockRange.toString())){
                            String curr_token = tokR.toString().substring(1, tokR.toString().length() - 1);
                            start_token = curr_token.split(",")[0];
                            found = 1;
                            break;
                        }
                        if (found == 1){
                            break;
                        }
                    }
                    if (found == 1){
                        break;
                    }
                    rangeInd++;
                }
                TokenRange toFind = metadata.newTokenRange(metadata.newToken(start_token), metadata.newToken(start_token));
                Set<Host> replicas = metadata.getReplicas(Metadata.quote(nodeKp), toFind);
                for (Host host : replicas) {
                    String finalHost = ((host.toString().split(":")[0]).split("/")[1]);
                    resultSet.add(finalHost);
                }
                return resultSet;
            }
        }
    }

    private static synchronized void checkCassandra() {
        if (cluster == null) {
            System.out.println(nodeIP);
            cluster = new Cluster.Builder()
                    .addContactPoints(nodeIP)
                    .withPort(nodePort)
                    .build();
            session = cluster.connect();
        }

    }

    private static synchronized void closeCassandra() {
        if (cluster != null) {
            session.close();
            session=null;
            cluster.close();
            cluster=null;
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
        closeCassandra();

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
        System.out.println("configFile:"+configFile);
        client = new StorageItf();

        try {
            client.init(configFile);
        } catch (StorageException e) {
            e.printStackTrace();
        }
        try {
            System.out.println(client.getLocations("21176a8c-b880-11e6-85e8-001517e69c14"));
            System.out.println(client.getLocations("Words_1"));
        } catch (StorageException e) {
            e.printStackTrace();
        }
        System.out.println("Application ended");
    }
}
