import pandas as pd
import numpy as np 
import psycopg2
import psycopg2.extras as extras
from app import clean_data
conn = psycopg2.connect( dbname='epilepsy',host='localhost',user='project',password='Password',port='5432')
curr= conn.cursor()

def db_init():
    df = pd.read_excel('Copy_epileptic_seizures_Responses_New.xlsx')
    df = clean_data(df)
    cols=list(df)
    # print(cols)
    sql=''' CREATE TABLE epilepsydata (
        caseno TEXT,
        patientage TEXT,
        gender TEXT,
        eventDuration TEXT,
        stereotypic TEXT,
        events_occur_time TEXT,
        wandering_headbanging_observation TEXT,
        eyes_closed_during_event TEXT,
        weeping_before_during_after_episode TEXT,
        patient_fall_suddenly_without_limb_movements TEXT,
        Was_patient_hyperventilating TEXT,
        loss_consciousness_after_urination_defecation TEXT,
        side_to_side_head_nodding_pelvic_thrusting_Opisthotonic_posturing TEXT,
        observation_limbjerking TEXT,
        observe_postevent_stridulous_laboured_breathing TEXT,
        upper_limb_jerks_observation TEXT,
        brieflossoftouch TEXT,
        fixed_Aura_Premonition TEXT,
        staring_blankly_chewing_smacking_lips TEXT,
        posturing_limbeyehead_deviation TEXT,
        violent_thrashing_movements TEXT,
        recovery TEXT,
        bittentongue TEXT,
        urine_without_knowledge TEXT,
        dislocated_shoulder TEXT,
        other_injuries TEXT,
        final_diagnosis TEXT
    );'''

    curr.execute(sql)
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    
    # SQL query to execute
    table='epilepsydata'
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    try:
        extras.execute_values(curr, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
db_init()
conn.commit()
curr.close()
conn.close()
