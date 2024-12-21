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

    def query_data(self, table, where_conditions, fields):
        where_clause = " AND ".join([f"{key}='{value}'" for key, value in where_conditions.items()])
        query = f"SELECT {', '.join(fields)} FROM {table} WHERE {where_clause};"
        results = self.session.execute(query)
        return results.all()

    def get_all_entities(self, table):
        try:
            query = f"SELECT * FROM {table};"
            results = self.session.execute(query)

            return [row._asdict() for row in results]
        except Exception as e:
            print(f"Error al recuperar las entidades de la tabla {table}: {e}")
            return []
