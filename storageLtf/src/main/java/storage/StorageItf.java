package storage;

import com.datastax.driver.core.*;

import java.util.*;
import java.util.function.Function;

import static java.util.stream.Collectors.*;


public class StorageItf {

    private static Cluster cluster = null;
    private static Session session = null;



    /**
     * This function returns a order list of the hosts that have the higher number of local tokens.
     *
     * @param objectID the block identifier
     * @return
     * @throws storage.StorageException
     */
    public static List<String> getLocations(String objectID) throws storage.StorageException {
         checkCassandra();
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
            UUID uuid = UUID.fromString(objectID);
            String name = session.execute("SELECT name FROM hecuba.istorage WHERE storage_id = ?", uuid)
                    .one().getString("name");
            int pposition = name.indexOf('.');
            if (pposition == -1) {
                throw new StorageException("I cannot detect the keyspace name from " + name);
            }
            final String nodeKp = name.substring(0, pposition);

            Set<Map.Entry<Host, Long>> hostsTkns = session.execute("SELECT tokens FROM hecuba.istorage WHERE storage_id = ?", uuid)
                    .one().getList("tokens", TupleValue.class).stream()
                    .map(tok -> metadata.newToken(tok.getLong(0) + ""))
                    .flatMap(token ->
                            metadata.getReplicas(Metadata.quote(nodeKp), metadata.newTokenRange(token, token)).stream())
                    .collect(groupingBy(Function.identity(), counting())).entrySet();

            ArrayList<Map.Entry<Host, Long>> result = new ArrayList<>(hostsTkns);
            Collections.sort(result, Comparator.comparing(o -> (o.getValue())));
            List<String> toReturn;
            toReturn = result.stream().map(a -> a.getKey().getAddress().toString().replaceAll("^.*/", "")).collect(toList());
            closeCassandra();
            return toReturn;
        }
    }


    private static void checkCassandra() {
        if (cluster == null) {
            String[] nodeIP = System.getenv("CONTACT_NAMES").split(",");
            int nodePort = Integer.parseInt(System.getenv("NODE_PORT"));
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
        closeCassandra();

    }

    public static java.lang.Object getByID(String objectID) throws StorageException {
        return null;
    }

    public static void init(String configFile) throws storage.StorageException {

    }

    public static void main(String[] args) throws StorageException {

        StorageItf client = new StorageItf();

        try {
            client.init(null);
        } catch (StorageException e) {
            e.printStackTrace();
        }

        client.getLocations(args[0]).forEach(System.out::println);
        System.out.println("Application ended");
    }
}
