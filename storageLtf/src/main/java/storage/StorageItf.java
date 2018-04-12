package storage;
import com.datastax.driver.core.*;
import com.google.common.net.InetAddresses;
import java.util.*;
import java.util.function.Function;
import static java.util.stream.Collectors.*;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.regex.Pattern;

public class StorageItf {

    private static Cluster cluster = null;
    private static Session session = null;
    private static Map<String, String> resolution = null;

    /**
     * This function returns a order list of the hosts that have the higher number of local tokens.
     *
     * @param storageId the block identifier
     * @return
     * @throws storage.StorageException
     */
    public static List<String> getLocations(String storageId) throws storage.StorageException {
 UUID uuid;
		   int nodePort;
		   String [] nodeIP = null;
		   //System.out.println("storageId: "+storageId);
		   Set<String> IPToReturn = new HashSet<String>();
		   Set<String> ToReturn = new HashSet<String>();
		   List<String> list = null;
		   String nodesVar;
		   Row storage_info = null;

		   if (storageId == null) {
			throw new StorageException("@getLocations: received a null storageId");
		   }
		   
		   if (storageId.contains(" ")){
			   // is this possible???
			   uuid = UUID.fromString(storageId.replace(" ", ""));
			   System.out.println("DEBUG: storage id contains blank spaces");
		   } else {	
			   uuid = UUID.fromString(storageId);
		   }
		   // TEMP: TO DEAL WITH DISJOINT SET OF NODES
		   nodesVar=System.getenv("PYCOMPSS_NODES");	
		   if (nodesVar != null) {
			   String[] PyCOMPSsNodes=nodesVar.split(",");	   	  	   
			   for (String node: PyCOMPSsNodes){
				   ToReturn.add(node);
			   }
			   list=new ArrayList<String>(ToReturn);
		   	   //System.out.println("@Hecuba getLocation: storageId - " +storageId + " " +  list.toString());
			   return list;
		   } else {
		   	   //System.out.println("@Hecuba getLocation: PYCOMPSS_NODES is not set");
		   }
		 	
		   // END TEMP
		   nodesVar=System.getenv("CONTACT_NAMES");
		   if (nodesVar==null){		   			   
			   nodeIP = new String[1];
			   nodeIP[0]="localhost";
		   } else {
			   nodeIP = nodesVar.split(",");
		   }
		   String port =System.getenv("NODE_PORT");
		   if (port == null)
			   nodePort=9042;
		   else         	          
			   nodePort = Integer.parseInt(port);               
		   if (resolution == null) {   
			   resolution = new HashMap<String, String>();
			   for (String toResolve:nodeIP){
				   try {
					   if (!InetAddresses.isInetAddress(toResolve)) {
						   String resolved=InetAddress.getAllByName(toResolve)[0].toString().split("/")[1];
						   resolution.put(resolved,toResolve);
						   System.out.println("@Hecuba getLocation: Resolution[ "+ resolved +"]="+toResolve);
					   } else {
						   resolution.put(toResolve,toResolve);
					   }
				   } catch (UnknownHostException e) {
					   // TODO Auto-generated catch block
					   e.printStackTrace();
				   }            			
			   }

		   }		   
		   if (cluster == null) {                
			   try {
				   cluster = new Cluster.Builder()
						   .addContactPoints(nodeIP[0])
						   .withPort(nodePort)
						   .build();

				   session = cluster.connect();
			   } catch (Exception e){
				   System.out.println("@getLocations: Exception connecting to Cassandra " + e.toString());
				   e.printStackTrace();
			   }
		   }
		   try {

		   	storage_info = session.execute("SELECT tokens,name FROM hecuba.istorage WHERE storage_id = ?", uuid).one();	            
		   } catch (Exception e) {
			System.out.println("@getLocations: Exception selecting tokens for " + storageId + " "+ e.toString());
			e.printStackTrace();

		   }

		   //System.out.println("@Hecuba getLocation: storage_info: " + storage_info.toString());

		   List<TupleValue> tokenList = storage_info.getList("tokens", TupleValue.class);
		   
		   String [] name = storage_info.getString("name").split("\\."); //first field is the keyspace name, the rest the table name
		   if (name.length <=1) {
			throw new StorageException("@getLocations: field name in istorage table has wrong format: " + storage_info.getString("name"));
		   }
		   String ks=name[0]; 
		   
		   //System.out.println("@Hecuba getLocation list of tokens: " + tokenList.toString());

		   Metadata metadata = cluster.getMetadata();

		   
		   for (TupleValue token_tuple: tokenList){
			   Set <Host> host_set;
			   TokenRange tr;
			   //Create a TokenRange object
			   Token token_init = metadata.newToken(Long.toString(token_tuple.getLong(0)));
			   Token token_end = metadata.newToken(Long.toString(token_tuple.getLong(1)));
			   tr=metadata.newTokenRange(token_init,token_end);
			   host_set=metadata.getReplicas(ks, tr);
			   //System.out.println("@Hecuba getLocations keyspace: " + ks );
			   for (Host h:host_set){
				   //System.out.println("@Hecuba getLocations hosts: " + h.getAddress().toString());
				   String ip=h.getAddress().toString().split("/")[1];
				   IPToReturn.add(ip);	
				   ToReturn.add(resolution.get(ip));	 

			   }
		   }

		   //System.out.println(IPToReturn.toString());


		   //cluster.close();
		   //cluster=null;
		   
		   list = new ArrayList<String>(ToReturn);
		   //System.out.println("@Hecuba getLocation: storageId - " +storageId + " " +  list.toString());

		   return list;

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

    public static String newVersion(String objectID, boolean preserveSrc, String node) throws StorageException {
        //return "";
        return objectID;
    }

    public static void consolidateVersion(String objectID) throws StorageException {
    }

    // java interface with PyCOMPSs does not need this function
    //public static void delete(String objectID) throws StorageException {
    //}

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
