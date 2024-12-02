from cassandra.cluster import Cluster


class Cassandra:
    def __init__(self, hosts, keyspace):
        self.cluster = Cluster(hosts)
        self.session = self.cluster.connect()
        self.session.set_keyspace(keyspace)

    def insert_data(self, table, data):
        keys = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        query = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
        self.session.execute(query, tuple(data.values()))

    def close(self):
        self.cluster.shutdown()
