import vector
import iris


# Open a connection to the server
args = {
	'hostname':'127.0.0.1', 
	'port': 51773,
	'namespace':'USER', 
	'username':'superuser', 
	'password':'sys'
}
conn = iris.connect(**args)
store = vector.IrisVectorStore(connection=conn)
store.initialise_schema()
# Create an iris object
# irispy = iris.createIRIS(conn)

# Create a global array in the USER namespace on the server
# irispy.set("^myGlobal", "hello world!") 

# Read the value from the database and print it
# print(irispy.get("myGlobal"))

# Delete the global array and terminate
# irispy.kill("myGlobal") 
conn.close()
