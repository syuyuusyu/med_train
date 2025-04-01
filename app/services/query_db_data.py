import pymysql
from pymysql import Error
from .dbClient import DbClient


def doc_info():
    from app import container
    config = container.config.datasource
    hpMap = container.config.hpNames
    with DbClient(config) as db_client:

        sql = f"select hp_id,his_id,name,title,good_at,info from his_doctor where hp_id in ('415') and good_at is not null and info is not null"
        results = db_client.query(sql)
        passages = []
        for result in results:
            depts = db_client.query("select hp_id,dept_id,name from his_dept where current_doc_ids like %s", (f"%{result['his_id']}%"))
            if len(depts) > 0:
            
                deptNames =  [dept['name'] for dept in depts]
                deptIds =  [dept['dept_id'] for dept in depts]
                deptIds = ','.join(deptIds)
                deptNames = ','.join(deptNames)
                hpName = hpMap.get(depts[0]['hp_id'], '未知医院')
            else:
                deptIds = ''
                deptNames = '医生当前没有排班'
                hpName = hpMap.get(result['hp_id'], '未知医院')
            passage = f'医院:{hpName}\n医院ID:{result["hp_id"]}\n姓名:{result["name"]}\n医生ID:{result["his_id"]}\n{result["title"]}\n{result["good_at"]}\n{result["info"]}\n所在科室:{deptNames}\n所在科室ID:{deptIds}\n'
            #print(passage)
            passages.append(passage)
        return passages

