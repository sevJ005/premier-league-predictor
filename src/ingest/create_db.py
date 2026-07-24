import sqlite3

# connecting to database file
connection = sqlite3.connect("db/pl_data.db")

# cursor object and execution
cusror = connection.cursor()

with open("db/schema.sql", "r") as f:
    schema_sql = f.read()

cusror.executescript(schema_sql)

connection.commit()
connection.close()
