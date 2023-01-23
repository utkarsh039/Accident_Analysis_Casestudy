from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import desc, count, col, dense_rank
def num_crashes_male(spark):
    df_pp = spark.read.csv("Data/Primary_Person_use.csv", header=True)
    df_units = spark.read.csv("Data/Units_use.csv", header=True)
    num_crashes_male_cnt = df_pp.filter('PRSN_INJRY_SEV_ID="KILLED" AND PRSN_GNDR_ID="MALE"') \
                   .select('CRASH_ID').distinct().count()
    print('Number of crashes(accidents) in which number of persons killed are male:', num_crashes_male_cnt)
    return num_crashes_male_cnt

def two_wheelers_crashes(spark):
    df_units = spark.read.csv("Data/Units_use.csv", header=True)
    two_wheeler_vehicle_body = ['POLICE MOTORCYCLE', 'MOTORCYCLE']
    df_units_two_wheel = df_units.select('CRASH_ID').where(df_units.VEH_BODY_STYL_ID.isin(two_wheeler_vehicle_body))
    df_units_two_wheel_count = df_units_two_wheel.distinct().count()

    print('Number of two wheelers which are booked for crashes:', df_units_two_wheel_count)
    return df_units_two_wheel_count

def num_crashes_female(spark):
    df_pp = spark.read.csv("Data/Primary_Person_use.csv", header=True)
    # Filter gender for FEMALE on PRSN_GNDR_ID and getting groupped on DRVR_LIC_STATE_ID desc to get highest number of accident
    num_crash = df_pp.select("PRSN_GNDR_ID", "DRVR_LIC_STATE_ID").filter(df_pp.PRSN_GNDR_ID == "FEMALE")\
        .groupby("DRVR_LIC_STATE_ID").count().orderBy(col("count").desc()).first().DRVR_LIC_STATE_ID
    print('Number of crashes(accidents) in which number of persons are female: ',num_crash)
    return num_crash

def top_5th_15th_make(spark):
    df_pp = spark.read.csv("Data/Primary_Person_use.csv", header=True)
    df_units = spark.read.csv("Data/Units_use.csv", header=True)
    injury_flag_list = ['KILLED', 'NON-INCAPACITATING INJURY', 'POSSIBLE INJURY', 'INCAPACITATING INJURY']

    # Joining Units dataset with rimary person dataset to filter only the cashes involving injuries 
    person_units_inner_join_df = df_pp.filter(col('PRSN_INJRY_SEV_ID').isin(injury_flag_list)) \
                                .select('CRASH_ID', 'UNIT_NBR') \
                                .join(df_units.select(['CRASH_ID', 'UNIT_NBR', 'VEH_MAKE_ID']), ['CRASH_ID', 'UNIT_NBR'], 'inner')

    # Counting unique crashes fo each VEH_MAKE_ID(Eliminating cases where VEH_MAKE_ID is NA)
    person_veh_make_df = person_units_inner_join_df.select('CRASH_ID', 'VEH_MAKE_ID').filter('VEH_MAKE_ID!="NA"') \
                        .distinct().groupby('VEH_MAKE_ID').count().orderBy(desc('count'))

    # Assiging rank based on the crashes/count of each vehicle make
    windowSpec = Window.orderBy(desc('count'))
    person_veh_make_df = person_veh_make_df.withColumn('dense_rank', dense_rank().over(windowSpec))

    # Filtering top 5th to 15th VEH_MKE_IDs
    print('Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death')
    person_veh_make_df.filter('dense_rank>=5 AND dense_rank<=15').select(['VEH_MAKE_ID']).show(truncate=False)

def top_ethnic_gp(spark):
    df_pp = spark.read.csv("Data/Primary_Person_use.csv", header=True)
    df_units = spark.read.csv("Data/Units_use.csv", header=True)

    person_units_join_df = df_pp.join(df_units, ['CRASH_ID', 'UNIT_NBR'], 'outer')
    body_ethnic_df = person_units_join_df.select(['PRSN_ETHNICITY_ID', 'VEH_BODY_STYL_ID', 'CRASH_ID']) \
                    .filter('VEH_BODY_STYL_ID!="NA" AND VEH_BODY_STYL_ID!="UNKNOWN"').distinct() \
                    .groupby(['VEH_BODY_STYL_ID','PRSN_ETHNICITY_ID']) \
                    .agg(count('CRASH_ID').alias('crash_count'))

    # Using dense rank to find the top ethnicity for each vehicle body styles
    windowSpec = Window.partitionBy('VEH_BODY_STYL_ID').orderBy(desc('crash_count'))
    body_ethnic_df = body_ethnic_df.withColumn('dense_rank', dense_rank().over(windowSpec))
    
    print(' top ethnic user group of each unique body style')
    body_ethnic_df.filter('dense_rank=1').select(['VEH_BODY_STYL_ID','PRSN_ETHNICITY_ID']) \
                    .orderBy('VEH_BODY_STYL_ID','PRSN_ETHNICITY_ID').show(truncate=False)

def top5_zip_crash(spark):
    df_units = spark.read.csv("Data/Units_use.csv", header=True)
    df_pp = spark.read.csv("Data/Primary_Person_use.csv", header=True)
    # Using contributing factor and dropping null values
    df = df_units.join(df_pp, on=['CRASH_ID'], how='inner').\
    dropna(subset=["DRVR_ZIP"]).\
    filter(col("CONTRIB_FACTR_1_ID").contains("ALCOHOL") | col("CONTRIB_FACTR_2_ID").contains("ALCOHOL")).\
    groupby("DRVR_ZIP").count().orderBy(col("count").desc()).limit(5)
    print('Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash')
    df.show(truncate = False)

def damage_above_4(spark):
    df_damages = spark.read.csv("Data/Damages_use.csv", header=True)
    df_units = spark.read.csv("Data/Units_use.csv", header=True)
    # Damage id above damage level 4
    damage_id = ['DAMAGED 5', 'DAMAGED 6', 'DAMAGED 7 HIGHEST']
    # Considering PROOF OF LIABILITY INSURANCE is filter for insurance
    count_of_crashid = df_damages.join(df_units, on=["CRASH_ID"], how='inner')\
    .filter((df_units.VEH_DMAG_SCL_1_ID.isin(damage_id))  | (df_units.VEH_DMAG_SCL_2_ID.isin(damage_id)))\
    .filter(df_damages.DAMAGED_PROPERTY == "NONE")\
    .filter(col("FIN_RESP_TYPE_ID").contains("PROOF OF LIABILITY INSURANCE")).\
    select("CRASH_ID", "VEH_DMAG_SCL_1_ID", "VEH_DMAG_SCL_2_ID", "FIN_RESP_TYPE_ID", "DAMAGED_PROPERTY").count()

    print('Count of Crash IDs where No Damaged property was observed and Damage Level is above 4 and car avails Insurance: ',count_of_crashid)
    return count_of_crashid

def top5_vehicle_makes(spark):
    df_charges = spark.read.csv("Data/Charges_use.csv", header=True)
    df_units = spark.read.csv("Data/Units_use.csv", header=True)
    df_pp = spark.read.csv("Data/Primary_Person_use.csv", header=True)
    # Filtering DRVR_LIC_TYPE_ID on licensed drivers
    # Filtering charge for speed
    # charge join to units for make_id
    
    # Getting top 25 states and top 1- vehicle color
    top_25_states = [row[0] for row in df_units.na.drop(subset=["VEH_LIC_STATE_ID"]).groupby("VEH_LIC_STATE_ID").count().orderBy(col("count").desc()).limit(25).collect()]

    top_10_vehicle_colors = [row[0] for row in df_units.filter(df_units.VEH_COLOR_ID != "NA").groupby("VEH_COLOR_ID").count().orderBy(col("count").desc()).limit(10).collect()]
    print('Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25 states')
    df_charges.join(df_pp, on=['CRASH_ID'], how='inner').\
    join(df_units, on=['CRASH_ID'], how='inner'). \
    filter(df_charges.CHARGE.contains("SPEED")). \
    filter(df_pp.DRVR_LIC_TYPE_ID.isin(["DRIVER LICENSE", "COMMERCIAL DRIVER LIC."])).\
    filter(df_units.VEH_COLOR_ID.isin(top_10_vehicle_colors)).\
    filter(df_units.VEH_LIC_STATE_ID.isin(top_25_states)).\
    groupby("VEH_MAKE_ID").count().\
    orderBy(col("count").desc()).show(5, truncate = False)


