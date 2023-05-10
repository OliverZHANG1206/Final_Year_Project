import csv
import sqlite3


class Database:
    def __init__(self):
        self.conn = sqlite3.connect("Data.db", check_same_thread=False)
        self.cur = self.conn.cursor()

    def create_table(self, table):
        cmd = "CREATE TABLE IF NOT EXISTS " + table + "\n" \
                                                      '''(ID           INTEGER  PRIMARY KEY AUTOINCREMENT,
                TIMESTAMP    DATETIME NOT NULL,
                CO2          REAL     NOT NULL,
                TEMPERATURE  REAL     NOT NULL,
                HUMIDITY     REAL     NOT NULL,
                PIR_TRIGGER  INT      NOT NULL);'''
        self.cur.execute(cmd)
        print('Create Table ' + table + ' Successfully')

    def update_data(self, table, content):
        cmd = "INSERT INTO " + table + " (Timestamp, CO2, Temperature, Humidity, PIR_trigger) VALUES(datetime('now','+8 hours'), " \
              + content[0] + ", " + content[1] + ", " + content[2] + ", " + content[3] + ");"
        self.cur.execute(cmd)
        self.conn.commit()
        print('Successfully upload to database')

    def save_csv(self, filename, table, stamp):
        cmd = "SELECT * FROM " + table + " WHERE ID > " + str(stamp) + ";"
        self.cur.execute(cmd)
        with open(filename, mode="w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            column_name = [tuple[0] for tuple in self.cur.description]
            writer.writerow(column_name)
            for row in self.cur.fetchall():
                writer.writerow(row)
            file.close()

    def current_stamp(self, table):
        cmd = "SELECT MAX(ID) FROM " + table + ";"
        self.cur.execute(cmd)
        stamp = self.cur.fetchall()[0][0]
        return stamp

    def check_table_exist(self, table):
        cmd = "SELECT count(*) FROM sqlite_master WHERE type=\"table\" AND name = \"" + table + "\";"
        self.cur.execute(cmd)
        if self.cur.fetchall()[0][0] > 0:
            return True
        else:
            return False

    def clear_table(self, table):
        cmd = "DELETE FROM " + table
        self.cur.execute(cmd)
        cmd = "UPDATE sqlite_sequence SET seq = 0 where name ='" + table + "';"
        self.cur.execute(cmd)
        self.conn.commit()
        print('Clear table successfully')
