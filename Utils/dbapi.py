# -*- coding:utf-8 -*-
import sqlite3

def delLastChar(value):
    tmp = list(value)
    tmp.pop()
    return ''.join(tmp)


class dbShell:

    def __init__(self, dbname):
        self.connection = sqlite3.connect(dbname)
        self.cursor = self.connection.cursor()
        pass

    def select(self, table='', cols='*', condition='', orderby=''):
        if table:
            table_name = table
            select_cols = cols
            select_sql = "select %s from %s " % (select_cols, table_name)

            if condition:
                select_conditon = condition
                select_sql += "where %s" % select_conditon

            if orderby:
                order = orderby
                select_sql += "order by %s" % (order)

            select_result = self.cursor.execute(select_sql).fetchall()

            return select_result
        else:
            return ''

    def insert(self, table=None, cols=None, value=None):
        if table is not None:
            table_name = str(table)
            cols_nums = value[0].__len__()
            if cols is not None:
                insert_cols = str(cols)
                insert_sql = "insert into %s(%s) VALUES (%s)" % (table_name, insert_cols, delLastChar('?,' * cols_nums))
            else:
                insert_sql = "insert into %s VALUES (%s)" % (table_name, delLastChar('?,' * cols_nums))
            self.connection.cursor().executemany(insert_sql, value)
            self.connection.commit()
        else:
            return False
        pass

    def delete(self, table=None, condition=None):
        if table is not None and condition is not None:
            delete_sql = "DELETE FROM %s WHERE %s" % (table, condition)
            self.cursor.execute(delete_sql)
            self.connection.commit()

    def delete_all(self, table=None):
        if table is not None:
            delete_sql = "DELETE FROM %s" % (table)
            self.cursor.execute(delete_sql)
            self.connection.commit()

    def count(self, table=None, condition=None):
        if table is not None:
            count_sql = "select count(*) from %s" % (table)
            if condition is not None:
                count_sql += ' where %s' % condition
            count_result = self.cursor.execute(count_sql).fetchone()[0]
            return count_result

    def close(self):
        self.connection.cursor().close()
