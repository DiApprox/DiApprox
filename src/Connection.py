import psycopg2

class Connection:

    def __init__(self):
        self.conn =psycopg2.connect(database='', user='', password='', host='127.0.0.1', port= '5432')

    
    def __del__(self):
        self.conn.commit()
        self.conn.close()



