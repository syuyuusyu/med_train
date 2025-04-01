import pymysql
from pymysql.cursors import DictCursor

class DbClient:
    def __init__(self, config):
        """
        初始化数据库连接。
        
        Args:
            host (str): 数据库主机地址
            user (str): 数据库用户名
            password (str): 数据库密码
            database (str): 数据库名称
            port (int): 数据库端口，默认为 3306
            charset (str): 字符集，默认为 utf8mb4
        """
        try:
            self.connection = pymysql.connect(
                host=config['host'],
                user=config['user'],
                password=config['password'],
                database=config['database'],
                port=config.get('port', 3306),
                charset='utf8mb4',
                cursorclass=DictCursor
            )
        except pymysql.Error as e:
            raise Exception(f"Failed to connect to database: {str(e)}")

    def query(self, sql, params=None):
        """
        执行 SQL 查询并返回结果。
        
        Args:
            sql (str): SQL 查询语句
            params (tuple or dict, optional): 查询参数，用于防止 SQL 注入
            
        Returns:
            list: 查询结果（字典列表）
            
        Raises:
            Exception: 如果查询执行失败
        """
        cursor = None
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except pymysql.Error as e:
            raise Exception(f"Failed to execute query: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def close(self):
        """
        关闭数据库连接。
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        """
        支持上下文管理器（with 语句）。
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        确保在上下文管理器退出时关闭连接。
        """
        self.close()
