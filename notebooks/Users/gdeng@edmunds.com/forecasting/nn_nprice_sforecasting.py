# Databricks notebook source
# MAGIC %md
# MAGIC #  NN (tensorflow) model for forecasting new vehicle prices
# MAGIC ## run weekday to catch any transdata updates and any new vehicle entries
# MAGIC ## Forecasting price of each run can be used for daily/weekly/monthly

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * traindata: summarized transaction data by style, salemonth whenever data being updated
# MAGIC * forecasting data: Edmunds new vehicles listed
# MAGIC * create age for all foecasted vehicles (age:from 1-36 month), the brand new vehicle's age will be automatically adjusted based on actual sales data later on
# MAGIC * All features of the forecasting data derived from prior month/two month/three month, and some from couple of years data based on sales counts and time,age,style,model,category,make
# MAGIC * If no features can be assiged, for example, in May, 2020,only few sales for 2018 models, so most 2018 model price borrowed from long term forecasting or unnecessary forecasting with assigned  depreciation rate for 2018 models forcasted by the end of 2019
# MAGIC * exotic vehicles (category='R') wont be forecasted, all prices =msrp
# MAGIC * The forcasted price will be converted margin (forecasted price/msrp_applied), hoever, dealer cash impact will be adjusted if the dealer cash is so different from NN model used
# MAGIC * NN model mainly to catch seasonal /monthly impact on model/category/make level
# MAGIC * the NN modeing (with L2 elastic net function imbeded) is fully automated,powerful, and self optimized  process as time by for forecasting month with daily run

# COMMAND ----------



# COMMAND ----------

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
import sys
import os
import matplotlib as pt
import pyspark.sql.functions as F
from pyspark.sql import Window as W
from pyspark.sql.types import FloatType
from datetime import datetime
from datetime import timedelta
from datetime import date
import sklearn
import pyspark.sql.types as T



# COMMAND ----------


print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + pt.__version__)
print('tensorflow version ' + tf.__version__)
print('numpy version ' +  np.__version__)
print('spark version ' +spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"))



# COMMAND ----------

from pyspark.sql.functions import pandas_udf,PandasUDFType
@pandas_udf("float",PandasUDFType.SCALAR)
def ptgsharepandas_udf(price,msrp):
  return round(price/msrp,4)
def age_year(saleyear,year):
  return(F.when(year==saleyear,0).when(year==saleyear-1,1).when(year==saleyear-2,2) \
   .when(year==saleyear+1,-1).when(year==saleyear+2,-2).otherwise(10))
from pyspark.sql.functions import udf
@udf("float")
def ptgshare_udf(price,msrp):
  return round(price/msrp,4)
def switch(x1,x2):
  return(F.when(x1>=0,x1).otherwise(x2))


# COMMAND ----------

def toAssignStartMM():
  data1=spark.sql("SELECT year_of_sale*100+ month_of_sale as syyyymm, year, make, model, trans_count from stats.try_transdatayrmd where year>=year(current_date)-2 and trans_count>20")
  data11=spark.sql("SELECT year_of_sale*100+ month_of_sale as syyyymm, year, make, model, trans_count from stats.try_transdatayrmd where year>=year(current_date)-2 and trans_count>2")
  cat=W.partitionBy(data1.year,data1.make,data1.model).orderBy("syyyymm")
  data2=data1.withColumn("age",F.row_number().over(cat)).filter("age=1")
  cat=W.partitionBy(data11.year,data11.make,data11.model).orderBy("syyyymm")
  data22=data11.withColumn("age",F.row_number().over(cat)).filter("age=1")
  ncol=["year","make","model","yymminitial1","trans_count"]
  data3=data2.withColumnRenamed("syyyymm","yymminitial1").drop("age") \
        .withColumn("make",F.lower(F.col("make"))).withColumn("model",F.lower(F.col("model"))) \
        .dropDuplicates(subset=["year","make","model","yymminitial1"]).select(*ncol)
  data33=data22.withColumnRenamed("syyyymm","yymminitial1").drop("age") \
        .withColumn("make",F.lower(F.col("make"))).withColumn("model",F.lower(F.col("model"))) \
        .dropDuplicates(subset=["year","make","model","yymminitial1"]).select(*ncol)
  data4=data3.union(data33.join(data3,['year','make','model'],how='left_anti')).dropDuplicates(subset=["year","make","model"]).cache()
  newvehiclelistextra=spark.sql("select  ed_model_id,year,make,model from stats.try_nn_newvehiclesextraf    group by ed_model_id,year,make,model") \
                      .withColumn("year",F.col("year").cast(T.IntegerType())) \
                      .withColumn("make",F.lower(F.col("make"))).withColumn("model",F.lower(F.col("model"))).dropDuplicates(subset=["year","make","model"]) \
                      .withColumn('publishdate',F.year(F.lit(fdate))*100+F.month(F.lit(fdate)))
  newvehiclelist=spark.sql("select  ed_model_id,year,make,model, min(year(date_add(publish_date,17))*100+month(date_add(publish_date,17))) as publishdate \
                           from stats.try_nn_newvehiclesf group by ed_model_id,year,make,model").withColumn("year",F.col("year").cast(T.IntegerType()))  \
                      .withColumn("make",F.lower(F.col("make"))).withColumn("model",F.lower(F.col("model"))).dropDuplicates(subset=["year","make","model"]) 
  extra=newvehiclelistextra.join(newvehiclelist,['ed_model_id'],how='left_anti')
  newvehiclelist=newvehiclelist.union(extra)
  
  vehicle1=spark.sql("select * from stats.nn_vehicleagemaster").withColumn("year",F.col("year").cast(T.IntegerType()))
  col=["year","make","model","ed_model_id","yymminitial"]
  vehiclemore=newvehiclelist.join(vehicle1,['year','make','model'],how='left_anti').dropDuplicates(subset=["year","make","model"]) \
           .join(data4,['year','make','model'],how='left') \
           .withColumn("yymminitial", F.when((F.col("yymminitial1")<F.col("publishdate"))&  (F.col('trans_count')>20),F.col("yymminitial1")).when(F.isnull("publishdate"),F.col("yymminitial1")).otherwise(F.col("publishdate")))  \
           .withColumn("yymminitial", F.when(F.isnull("yymminitial"),date.today().year*100+date.today().month).otherwise(F.col("yymminitial"))) \
           .select(*col)
  vehicle2 =vehicle1.join(data4,['year','make','model'],how='left') \
      .withColumn("yymminitial", F.when((F.col("yymminitial1")<F.col("yymminitial"))&(F.col('trans_count')>20),F.col("yymminitial1")).otherwise(F.col("yymminitial"))) \
      .select(*col)

  vehiclemaster=vehicle2.union(vehiclemore).dropDuplicates(subset=["year","make","model"]) 
  vehiclemasterf=spark.sql("select * from stats.nn_vehicleagemasterbackupff").withColumn("year",F.col("year").cast(T.IntegerType())) 
  print(vehiclemasterf.count())
  vehiclemasterf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.nn_vehicleagemasterbackupf")
  print(vehiclemaster.count())
  vehiclemaster.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.nn_vehicleagemaster")
  if vehiclemaster.count()>1000:
    vehiclemaster.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.nn_vehicleagemasterbackupff")
  return
#toAssignStartMM()
def toGetRawData(maxmm,syymm,syear):
  new_df=spark.sql("""select * from stats.nn_transstyleprice where year_of_sale*100+month_of_sale<={maxmm} and year_of_sale*100+month_of_sale>={syymm} \
                 and vehicle_code>0 and year>={syear} and make!='Tesla'""".format(syear=syear,maxmm=maxmm,syymm=syymm))#catch all updated data from May 18 for daily run
  new_df=new_df.select([F.col(x).alias(x.lower()) for x in  new_df.columns]).withColumn('year',F.col('year').cast("integer"))
  tmvlist=spark.sql("select ed_style_id, year,make,model,tmv_category,model_year_link_code,model_level_link_code,msr_price from \
                    stats.try_nn_newvehiclesf where tmv_category!='R' and    make!='Tesla' and make!='Polestar'")
  tmvlistextra=spark.sql("select ed_style_id, year,make,model,tmv_category,null as model_year_link_code, null as model_level_link_code,total_original_msrp as msr_price from \
                    stats.try_nn_newvehiclesextraf where tmv_category!='R' and    make!='Tesla' and make!='Polestar'")  
  tmvlist=tmvlist.union(tmvlistextra).withColumn('year',F.col('year').cast("integer"))
  print(tmvlist.count())
  tmvlist.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.nn_shortterm_tmvlist")
  tmvlist=spark.sql("select * from stats.nn_shortterm_tmvlist")
  agemaster=spark.sql("select * from stats.nn_vehicleagemaster") 

  extralist=spark.sql("select * from stats.nn_shortterm_extraagelist")
  agemaster=agemaster.join(extralist, ['year','make','model'],how='left').withColumn("yymminitial",F.when(F.col("age1")>0,F.col('age1')).otherwise(F.col('yymminitial'))).drop('age1') \
         .withColumn("earliestmm",(F.col('year')-1)*100+2) \
         .withColumn("yymminitial",F.when(F.col("yymminitial")<F.col("earliestmm"),F.col('earliestmm')) \
         .otherwise(F.col('yymminitial'))).drop('earliestmm')
  catfinal=spark.sql("select * from stats.try_catfinal").cache()
  ed4list=spark.sql("select  link_make as make,link_model as model, max(ed4) as ed4 from stats.try_yrmdcatorigin where saleyear>2016 group by make,model") \
                      .withColumn("make",F.lower(F.col("make"))).withColumn("model",F.lower(F.col("model"))).dropDuplicates(subset=["make","model"]) \
                      .join(catfinal,['ed4'],how='left').cache()
  originlist=spark.sql("select  distinct link_make,origin from stats.try_yrmdcatorigin where saleyear>2016") \
           .withColumn("make",F.lower(F.col("link_make"))).drop('link_make').dropDuplicates(subset=["make"]).cache()
  transdata=new_df \
  .withColumnRenamed("vehicle_code","ed_style_id") \
  .withColumnRenamed("year_of_sale","saleyear") \
  .withColumnRenamed("month_of_sale","salemonth") \
  .withColumn("syrmm",F.col('saleyear')*100+F.col('salemonth'))\
  .withColumn("ageyear",age_year(F.col("saleyear"),F.col("year"))) \
  .withColumn("make",F.lower(F.col("make"))).withColumn("model",F.lower(F.col("model")))\
  .fillna(100,subset=['dtt']) \
  .withColumn("item",F.col("ed_style_id").cast(T.StringType())) \
  .join(agemaster,['year','make','model'],how='left') \
   .withColumn("ptg",F.round(F.col("trans_price")/F.col("trans_msrp"),5))  \
  .join(ed4list,["make","model"],how="left") \
  .join(originlist,["make"],how="left") \
  .join(tmvlist.select("ed_style_id", "tmv_category", "model_year_link_code","model_level_link_code","msr_price"),["ed_style_id"],how="left")\
  .withColumn("msrpf", F.when(F.col("trans_msrp")<F.col("msr_price"),F.col("msr_price")).otherwise(F.col("trans_msrp"))) \
  .withColumn("ptgf", F.when(F.col("ptg")>1.05,1.05).otherwise(F.col("ptg"))) \
  .withColumn("pricef", F.round(F.col("ptgf")*F.col("msrpf")))\
  .cache()
  vehiclelist=tmvlist \
  .withColumn("make",F.lower(F.col("make"))).withColumn("model",F.lower(F.col("model")))\
  .withColumn("item",F.col("ed_style_id").cast(T.StringType())) \
  .join(agemaster,['year','make','model'],how='left') \
  .join(ed4list,["make","model"],how="left") \
  .join(originlist,["make"],how="left") \
  .withColumn('yymmref',F.col("year")*100+F.lit(7)) \
  .withColumn("yymminitial",F.when(F.col("yymmref")<F.col("yymminitial"),F.col("yymmref")).otherwise(F.col("yymminitial"))) \
   .withColumn('iyear',F.col('yymminitial').substr(1,4).cast("integer")) \
   .withColumn('imonth',F.col('yymminitial').substr(5,6).cast("integer")) \
   .withColumn("idate",F.to_date(F.concat_ws("-","iyear","imonth",F.lit(1)))) \
   .withColumn("fdate",F.lit(fdate)) \
   .withColumn("age",F.months_between(F.col("fdate"),F.col('idate'))+1) \
   .withColumn("age",F.when(F.col('age')<1,1).otherwise(F.col('age')))\
   .withColumn("pdate",F.expr("add_months(fdate,0)")) \
   .withColumn("saleyear", F.year('pdate')) \
   .withColumn("salemonth", F.month('pdate'))\
   .withColumn("syrmm",F.col('saleyear')*100+F.col('salemonth'))\
   .withColumn("ageyear",age_year(F.col("saleyear"),F.col("year"))) \
   .cache()
  print(transdata.count())
  print(vehiclelist.count())
  return [transdata,vehiclelist]
#rawdata=toGetRawData(transmaxyymm,transminyymm)
def toProcessTransData():
  df1=rawdata[0].withColumn('yymmref',F.col("year")*100+F.lit(7)) \
   .withColumn("ed_model_id",F.col("ed_model_id").cast(T.StringType())) \
   .withColumn("yymminitial",F.when(F.col("yymmref")<F.col("yymminitial"),F.col("yymmref")).otherwise(F.col("yymminitial"))) \
   .withColumn('iyear',F.col('yymminitial').substr(1,4).cast("integer")) \
   .withColumn('imonth',F.col('yymminitial').substr(5,6).cast("integer")) \
   .withColumn("idate",F.to_date(F.concat_ws("-","iyear","imonth",F.lit(1)))) \
   .withColumn("adate",F.to_date(F.concat_ws("-","saleyear","salemonth",F.lit(1)))) \
   .withColumn("fdate",F.lit(fdate)) \
   .withColumn("age",F.months_between(F.col("adate"),F.col('idate'))+1) \
   .withColumn("age",F.when(F.col('age')<1,1).otherwise(F.col('age')))\
   .withColumn("agef",F.months_between(F.col("fdate"),F.col('idate'))+1) \
   .withColumn("agef",F.when(F.col('agef')<1,1).otherwise(F.col('agef')))\
   .withColumn("pdate",F.expr("add_months(fdate,0)")) \
   .withColumn("tdate",F.expr("add_months(fdate,-3)")) \
   .withColumn("syrmmt", F.year('tdate')*100+F.month('tdate')) \
   .withColumn("psaleyear", F.year('pdate')) \
   .withColumn("psalemonth", F.month('pdate'))\
   .cache()
  extraagelist=df1.filter("age<0  and trans_count>=3").groupBy('year','make','model').agg(F.min(F.col("syrmm")).alias('age1'))
  df1=df1.filter("age>0")
  #extraagelist%.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.nn_shortterm_extraagelist")
  print(extraagelist.count())
  return df1
def toProcessTestData():
  testdata1=df1.where('syrmm>syrmmt')
  cat=W.partitionBy(testdata1.item).orderBy(F.desc("syrmm"))
  testdata2=testdata1.withColumn("seq",F.row_number().over(cat)).cache()
  testdata3=testdata2.filter('seq=1').filter('trans_count>4').drop('seq')
  testdata4=testdata2.filter('seq in (1,2)').drop('seq')
  testdata6=testdata2.drop('seq')
  testdata5=testdata4.join(testdata3,['ed_style_id'],how='left_anti')
  summary1=testdata5.groupBy('ed_style_id') \
    .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgm'),
       F.round(F.sum(F.col('msrpf')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('msrpm'),
       F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dcm'),
       F.sum(F.col('trans_count')).alias('cntm'))

  testdata7=testdata5.dropDuplicates(subset=["ed_style_id"]).join(summary1,['ed_style_id'],how='left') \
   .withColumn("ptgf",F.col("ptgm")) \
   .withColumn("msrpf",F.col("msrpm")) \
   .withColumn("pricef",F.round(F.col("msrpf")*F.col("ptgf")))\
   .withColumn("dc",F.round(F.col("dcm")))\
   .withColumn("trans_count",F.round(F.col("cntm"))).drop('ptgm','msrpm','cntm','dcm').filter('trans_count>4').cache()
  testdata8=testdata6.join(testdata3,['ed_style_id'],how='left_anti').join(testdata7.select('ed_style_id'),['ed_style_id'],how='left_anti')

  summary2=testdata8.groupBy('ed_style_id') \
    .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgm'),
       F.round(F.sum(F.col('msrpf')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('msrpm'),
       F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dcm'),
       F.sum(F.col('trans_count')).alias('cntm'))
  testdata9=testdata8.dropDuplicates(subset=["ed_style_id"]).join(summary2,['ed_style_id'],how='left') \
   .withColumn("ptgf",F.col("ptgm")) \
   .withColumn("msrpf",F.col("msrpm")) \
   .withColumn("pricef",F.round(F.col("msrpf")*F.col("ptgf")))\
   .withColumn("dc",F.round(F.col("dcm")))\
   .withColumn("trans_count",F.round(F.col("cntm"))).drop('ptgm','msrpm','cntm','dcm').filter('trans_count>2').cache()
  testdatap=testdata3.union(testdata7).union(testdata9)\
   .withColumn("saleyear",F.col("psaleyear")) \
   .withColumn("salemonth",F.col("psalemonth")) \
   .withColumn("age",F.col("agef")) \
   .withColumn("syrmm",F.col("saleyear")*100+F.col("psalemonth")).filter(~F.isnull('ed_style_id')).cache()
  print(testdatap.count())
  return testdatap
  #df1=toProcessTransData()
  #testdatap=toProcessTestData()
def depnum(mm):
  return(F.when(mm==12,-0.001).when(mm==1,0.0005).otherwise(0.0))
def toSummaryData():
  dsummary1=testdatap.groupBy('ed_model_id') \
         .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgmean') , \
          F.round(F.sum(F.col('dtt')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dttmean') ,\
          F.round(F.sum(F.col('cc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("cc_mean"), \
          F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("dc_mean"),\
          F.round(F.mean(F.col('trans_count'))).alias("transcount_mean"))
  dsummary11=df1.groupBy('year','age','model_year_link_code') \
         .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgmean') , \
          F.round(F.sum(F.col('dtt')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dttmean') ,\
          F.round(F.sum(F.col('cc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("cc_mean"), \
          F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("dc_mean"),\
          F.round(F.mean(F.col('trans_count'))).alias("transcount_mean")).withColumn("year",F.col("year")+1) 
  dsummary12=df1.groupBy('year','age','model_level_link_code') \
         .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgmean') , \
          F.round(F.sum(F.col('dtt')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dttmean') ,\
          F.round(F.sum(F.col('cc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("cc_mean"), \
          F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("dc_mean"),\
          F.round(F.mean(F.col('trans_count'))).alias("transcount_mean")).withColumn("year",F.col("year")+1)            
  dsummary2=testdatap.groupBy('year','ed4','origin','age') \
         .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgmean') , \
          F.round(F.sum(F.col('dtt')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dttmean') ,\
          F.round(F.sum(F.col('cc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("cc_mean"), \
          F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("dc_mean"),\
          F.round(F.mean(F.col('trans_count'))).alias("transcount_mean"))

  dsummary3=testdatap.groupBy('year','make','age') \
         .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgmean') , \
          F.round(F.sum(F.col('dtt')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dttmean') ,\
          F.round(F.sum(F.col('cc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("cc_mean"), \
          F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("dc_mean"),\
          F.round(F.mean(F.col('trans_count'))).alias("transcount_mean")) 
  dsummary31=df1.groupBy('year','ed4','make','age') \
         .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgmean') , \
          F.round(F.sum(F.col('dtt')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dttmean') ,\
          F.round(F.sum(F.col('cc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("cc_mean"), \
          F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("dc_mean"),\
          F.round(F.mean(F.col('trans_count'))).alias("transcount_mean")).withColumn("year",F.col("year")+1)
  dsummary4=df1.groupBy('year','ed4','origin','age') \
         .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgmean') , \
          F.round(F.sum(F.col('dtt')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dttmean') ,\
          F.round(F.sum(F.col('cc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("cc_mean"), \
          F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("dc_mean"),\
          F.round(F.mean(F.col('trans_count'))).alias("transcount_mean")).withColumn("year",F.col("year")+1)
  dsummary5=df1.groupBy('year','make','age') \
         .agg(F.round(F.sum(F.col('ptgf')*F.col('trans_count'))/F.sum(F.col('trans_count')),4).alias('ptgmean') , \
          F.round(F.sum(F.col('dtt')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias('dttmean') ,\
          F.round(F.sum(F.col('cc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("cc_mean"), \
          F.round(F.sum(F.col('dc')*F.col('trans_count'))/F.sum(F.col('trans_count'))).alias("dc_mean"),\
          F.round(F.mean(F.col('trans_count'))).alias("transcount_mean")).withColumn("year",F.col("year")+1) 
  return [dsummary1,dsummary11,dsummary12,dsummary2,dsummary3,dsummary31,dsummary4,dsummary5]
#sumdata=toSummaryData()
def toGetTestData():
  dcol=['ptgmean','msrpf_mean','dc_mean','dc_mean','cc_mean','dttmean','transcount_mean','flag1']
  vehiclelistf=rawdata[1].join(testdatap.select('ed_style_id','ptgf','msrpf','pricef','dc','cc','dtt','trans_count').withColumn("flag",F.lit(1)),['ed_style_id'],how='left')\
             .join(sumdata[0].withColumn("flag1",F.lit(2)),['ed_model_id'],how='left')\
             .withColumn('ptgf',switch(F.col('ptgf'),F.col('ptgmean')))\
             .withColumn('dc',switch(F.col('dc'),F.col('dc_mean')))\
             .withColumn('cc',switch(F.col('cc'),F.col('cc_mean')))\
             .withColumn('dtt',switch(F.col('dtt'),F.col('dttmean')))\
             .withColumn('flag',switch(F.col('flag'),F.col('flag1')))\
             .drop(*dcol)\
             .join(sumdata[1].withColumn("flag1",F.lit(21)),['year','age','model_year_link_code'],how='left')\
             .withColumn('ptgf',switch(F.col('ptgf'),F.col('ptgmean')))\
             .withColumn('dc',switch(F.col('dc'),F.col('dc_mean')))\
             .withColumn('cc',switch(F.col('cc'),F.col('cc_mean')))\
             .withColumn('dtt',switch(F.col('dtt'),F.col('dttmean')))\
             .withColumn('flag',switch(F.col('flag'),F.col('flag1')))\
             .drop(*dcol)\
            .join(sumdata[2].withColumn("flag1",F.lit(22)),['year','age','model_level_link_code'],how='left')\
             .withColumn('ptgf',switch(F.col('ptgf'),F.col('ptgmean')))\
             .withColumn('dc',switch(F.col('dc'),F.col('dc_mean')))\
             .withColumn('cc',switch(F.col('cc'),F.col('cc_mean')))\
             .withColumn('dtt',switch(F.col('dtt'),F.col('dttmean')))\
             .withColumn('flag',switch(F.col('flag'),F.col('flag1')))\
             .drop(*dcol)\
             .join(sumdata[3].withColumn("flag1",F.lit(3)),['year','ed4','origin','age'],how='left')\
             .withColumn('ptgf',switch(F.col('ptgf'),F.col('ptgmean')))\
             .withColumn('dc',switch(F.col('dc'),F.col('dc_mean')))\
             .withColumn('cc',switch(F.col('cc'),F.col('cc_mean')))\
             .withColumn('dtt',switch(F.col('dtt'),F.col('dttmean')))\
             .withColumn('flag',switch(F.col('flag'),F.col('flag1')))\
             .drop(*dcol)\
             .join(sumdata[4].withColumn("flag1",F.lit(4)),['year','make','age'],how='left')\
             .withColumn('ptgf',switch(F.col('ptgf'),F.col('ptgmean')))\
             .withColumn('dc',switch(F.col('dc'),F.col('dc_mean')))\
             .withColumn('cc',switch(F.col('cc'),F.col('cc_mean')))\
             .withColumn('dtt',switch(F.col('dtt'),F.col('dttmean')))\
             .withColumn('flag',switch(F.col('flag'),F.col('flag1')))\
             .drop(*dcol)\
             .join(sumdata[5].withColumn("flag1",F.lit(41)),['year','ed4','make','age'],how='left')\
             .withColumn('ptgf',switch(F.col('ptgf'),F.col('ptgmean')))\
             .withColumn('dc',switch(F.col('dc'),F.col('dc_mean')))\
             .withColumn('cc',switch(F.col('cc'),F.col('cc_mean')))\
             .withColumn('dtt',switch(F.col('dtt'),F.col('dttmean')))\
             .withColumn('flag',switch(F.col('flag'),F.col('flag1')))\
             .drop(*dcol)\
             .join(sumdata[6].withColumn("flag1",F.lit(5)),['year','ed4','origin','age'],how='left')\
             .withColumn('ptgf',switch(F.col('ptgf'),F.col('ptgmean')))\
             .withColumn('dc',switch(F.col('dc'),F.col('dc_mean')))\
             .withColumn('cc',switch(F.col('cc'),F.col('cc_mean')))\
             .withColumn('dtt',switch(F.col('dtt'),F.col('dttmean')))\
             .withColumn('flag',switch(F.col('flag'),F.col('flag1')))\
             .drop(*dcol)\
             .join(sumdata[7].withColumn("flag1",F.lit(6)),['year','make','age'],how='left')\
             .withColumn('ptgf',switch(F.col('ptgf'),F.col('ptgmean')))\
             .withColumn('dc',switch(F.col('dc'),F.col('dc_mean')))\
             .withColumn('cc',switch(F.col('cc'),F.col('cc_mean')))\
             .withColumn('dtt',switch(F.col('dtt'),F.col('dttmean')))\
             .withColumn('flag',switch(F.col('flag'),F.col('flag1')))\
             .withColumn("msrpf",F.when(F.isnull("msrpf"),F.col('msr_price')).otherwise(F.col('msrpf')))\
             .withColumn("ptgf",F.col("ptgf")*(1+depnum(F.col('salemonth'))))\
             .withColumn("pricef",F.round((F.col("ptgf")*F.col('msrpf'))))\
             .drop(*dcol).cache()
  vehiclelistextra=vehiclelistf.filter(F.isnull('ptgf'))
  vehiclelistff=vehiclelistf.filter(~F.isnull('ptgf'))
  print(vehiclelistextra.count())
  print(vehiclelistff.count())
  print(vehiclelistf.count())
  return [vehiclelistff,vehiclelistextra]
#vehiclelistf=toGetTrainData()

# COMMAND ----------

def toGetNNData(nnlayer):
  if nnlayer=='md':
    testdatapp=vehiclelistf[0].withColumn('item',F.col("model_level_link_code"))
    df2=df1.withColumn('item',F.col("model_level_link_code"))
  elif nnlayer=='mk':
    testdatapp=vehiclelistf[0].withColumn('item',F.col("make"))
    df2=df1.withColumn('item',F.col("make"))
  elif nnlayer=='ed4':
    testdatapp=vehiclelistf[0].withColumn('item',F.col("ed4"))
    df2=df1.withColumn('item',F.col("ed4"))
  itemdata=testdatapp.select('item').distinct().join(df2.select('item').distinct(),['item'])
  droplist=[]
  itemlist =[i[0] for i in  itemdata.select("item").filter(~F.isnull('item')).distinct().rdd.collect() if i[0] not in droplist]
  cols=['item','ed_style_id','ed4','saleyear','salemonth','syrmm','year','make','model','age','ageyear','pricef','msrpf','ptgf','dc','dtt','cc','trans_count','ed_model_id',
      'model_level_link_code','model_year_link_code']
  categories =[i for i in  itemlist]
  exprs = [F.when(F.col("item") == category, 1).otherwise(0).alias(category) for category in categories]
#categories1 =[i for i in  modelcodelist]
#exprs1 = [F.when(F.col("model_level_link_code") == category1, 1).otherwise(0).alias(category1) for category1 in categories1]

  traindata=df2.filter(df2.item.isin(*itemlist)).select(*cols,*exprs).orderBy(['syrmm','ed_model_id','age']).toPandas()
  testdata=testdatapp.filter(testdatapp.item.isin(*itemlist)).select(*cols,*exprs).orderBy(['syrmm','ed_model_id','age']).toPandas()
  print(traindata.shape)
  print(testdata.shape)
  return  [traindata,testdata]

# COMMAND ----------

def norm(x,xx):
  return (x - xx['mean']) / xx['std']
def build_model(x):
  model = keras.Sequential([
    #layers.Dense(128, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001),activation='relu', input_shape=[len(x.keys())]),
    #layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001),activation='relu', input_shape=[263]),
    #layers.Dropout(0.2),
    layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001),activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(64,kernel_regularizer=keras.regularizers.l2(0.001),activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')  
def NNmodel(loops):
  dlist=['item','ed_style_id','ed4','syrmm','make','model','trans_count','ed_model_id','cc',
      'model_level_link_code','model_year_link_code','ageyear']
  train_dataset=datafset[0].drop(dlist,axis=1)
  test_dataset=datafset[1].drop(dlist,axis=1)
  train_dataset=train_dataset.astype(float)
  test_dataset=test_dataset.astype(float)
  train_stats = train_dataset.describe()
  train_stats.pop("pricef")
  train_stats = train_stats.transpose()
  train_labels = train_dataset.pop('pricef')
  test_labels = test_dataset.pop('pricef')
  normed_train_data = norm(train_dataset,train_stats)
  normed_test_data = norm(test_dataset,train_stats)
  model=build_model(train_dataset)
  history = model.fit(
  normed_train_data, train_labels,
  epochs=loops, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
  test_predictions = model.predict(normed_test_data).flatten()
  loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
  print("\n\n")
  print("Testing set Mean Abs Error\n\n: {:5.0f} $".format(mae))
  test_predictions = model.predict(normed_test_data).flatten()
  predictdata=pd.DataFrame(test_predictions,test_labels).reset_index().rename(columns={0:'predict'}).reset_index().drop(['index'],axis=1)
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  model.save('testmodel.h5')
  return [hist,predictdata]
def showValuationError():
  print(datasetf[0][['mean_absolute_error','val_mean_absolute_error','epoch']].tail(5))
  print(datasetf[1].sample(5))
  return;
#displaySample1() 
def f(x):
  d={}
  d["ptgdiff"]=round(abs(x.pricef/x.msrpf-x.predict/x.msrpf).mean(),4)
  d["mae_uw"]=round(abs(x.pricef-x.predict).mean())
  d["price_uw"]=round(x.pricef.mean())
  d["mae_w"]=round(abs((x.pricef-x.predict)*x.trans_count).sum()/x.trans_count.sum())
  d["price_w"]=round((x.pricef*x.trans_count).sum()/x.trans_count.sum())
  d["%mape_w"]=round(((abs(x.pricef-x.predict)/x.pricef)*x.trans_count).sum()/sum(x.trans_count),4)*100
  d["%rmspe_w"]=round(np.sqrt(((abs(x.pricef-x.predict)/x.pricef)**2*x.trans_count).sum()/sum(x.trans_count)),4)*100
  return pd.Series(d,index=['ptgdiff','mae_uw','price_uw','mae_w','price_w','%mape_w','%rmspe_w'])

def displayTurningData():
  predictset=pd.DataFrame(pd.concat([datasetf[1],datafset[1].reset_index().drop(['index'],axis=1).drop(['pricef'],axis=1)], axis=1))
  #spark.createDataFrame(predictset.groupby(['item']).apply(f).reset_index()).write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.try_nn_mmlist")
  #print(predictset.info())
  print(np.average(predictset.pricef,weights=predictset.trans_count))
  print(np.average(abs(predictset.pricef-predictset.predict),weights=predictset.trans_count))
  print(predictset.groupby(['item']).apply(f).reset_index().sample(5))
  print(predictset.groupby(['age']).apply(f).reset_index().sample(5))
  return
#displayData1()
def saveTodelta(nnlayer):
  cols=['ed_style_id','layer','item','age','msrpf','pricef','predict','ptg','year','make','model','ed4','flag']
  predictset=pd.DataFrame(pd.concat([datasetf[1],datafset[1].reset_index().drop(['index'],axis=1).drop(['pricef'],axis=1)], axis=1))
  #itemnew= [i[0] for i in spark.createDataFrame(predictset.groupby(['item']).apply(f).reset_index()).filter("ptgdiff<=0.02").select('item').distinct().rdd.collect()]
  pdataf=predictset.assign(ptg=round(predictset.predict/predictset.msrpf,4)).assign(flag='p').assign(layer=nnlayer)[cols]
  pdataf[['ed_style_id','age','msrpf','pricef','predict','year']]=pdataf[['ed_style_id','age','msrpf','pricef','predict','year']].astype(int)
  print(pdataf.shape)
  #if layer=='md':
  #spark.createDataFrame(pdataf).write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.nn_shortterm_data")
  #elif layer=='mk':
  print(spark.sql("""delete from stats.nn_shortterm_data  where layer='{nnlayer}'""".format(nnlayer=nnlayer)).count())
  spark.createDataFrame(pdataf).write.format("delta").mode("append").saveAsTable("stats.nn_shortterm_data")
  print(spark.sql("""select * from stats.nn_shortterm_data  where layer='{nnlayer}'""".format(nnlayer=nnlayer)).count())
  return

# COMMAND ----------


def toCreateForecastdata():
  col=['fdate','saleyear','salemonth','ed_style_id','age','year','make','model','ed_model_id','tmv_category','msr_price','msrpf','pricefinal','ptgfinal','ptgf','dc','cc','dtt','flag']
  predictmodels1=vehiclelistf[0] \
      .withColumn("item",F.col("model_level_link_code")) \
      .join(spark.sql("select ed_style_id,round(predict) as predict_md1 from stats.nn_shortterm_data \
                      where (round(predict)-pricef)/msrpf>-0.001 and (round(predict)-pricef)/msrpf<0.001 and layer='md'"),['ed_style_id'],how='left')\
      .join(spark.sql("select ed_style_id,round(predict) as predict_md2 from stats.nn_shortterm_data \
                      where (round(predict)-pricef)/msrpf>-0.002 and (round(predict)-pricef)/msrpf<0.002 and layer='md'"),['ed_style_id'],how='left')\
      .withColumn("item",F.col("ed4")) \
      .join(spark.sql("select ed_style_id,round(predict) as predict_ed1 from stats.nn_shortterm_data \
                      where (round(predict)-pricef)/msrpf>-0.001 and (round(predict)-pricef)/msrpf<0.001 and layer='ed4'"),['ed_style_id'],how='left')\
      .join(spark.sql("select ed_style_id,round(predict) as predict_ed2 from stats.nn_shortterm_data \
                      where (round(predict)-pricef)/msrpf>-0.002 and (round(predict)-pricef)/msrpf<0.002 and layer='ed4'"),['ed_style_id'],how='left')\
      .withColumn("item",F.col("make")) \
      .join(spark.sql("select ed_style_id,round(predict) as predict_mk1 from stats.nn_shortterm_data \
                      where (round(predict)-pricef)/msrpf>-0.001 and (round(predict)-pricef)/msrpf<0.001 and layer='mk'"),['ed_style_id'],how='left')\
      .join(spark.sql("select ed_style_id,round(predict) as predict_mk2 from stats.nn_shortterm_data \
                      where (round(predict)-pricef)/msrpf>-0.002 and (round(predict)-pricef)/msrpf<0.002 and layer='mk'"),['ed_style_id'],how='left')
  print(predictmodels1.count())
  predictmodels2=predictmodels1.withColumn("pricefinal",F.lit(None))\
               .withColumn("pricefinal",F.when(~F.isnull('predict_md1'),F.col('predict_md1'))\
               .when((F.isnull("pricefinal") & ~F.isnull('predict_ed1')),F.col('predict_ed1'))\
               .when((F.isnull("pricefinal") & ~F.isnull('predict_mk1')),F.col('predict_mk1'))\
               .when((F.isnull("pricefinal") & ~F.isnull('predict_md2')),F.col('predict_md2'))\
               .when((F.isnull("pricefinal") & ~F.isnull('predict_ed2')),F.col('predict_ed2'))\
               .when((F.isnull("pricefinal") & ~F.isnull('predict_mk2')),F.col('predict_mk2'))\
               .otherwise(F.col("pricef")))\
               .withColumn("ptgf",F.col("ptgf")/(1+depnum(F.col('salemonth'))))\
               .withColumn("ptgfinal",F.round(F.col("pricefinal")/F.col("msrpf"),4)).select(*col).cache()
  predictmodels3=vehiclelistf[1].join(spark.sql("""select ed_style_id,ptg from stats.nn_longtermforecast_finalhist \
                                               where fdate=to_date('{fdate}') and pdate=fdate""".format(fdate=fdate)),['ed_style_id'],how='left')\
                .withColumnRenamed("ptg","ptgfinal")\
                .withColumn("pricefinal",F.round(F.col('ptgfinal')*F.col('msrpf')))\
                .withColumn("flag",F.lit(99)).select(*col).cache()
  predictmodels=predictmodels2.union(predictmodels3)
  print(fdate)
  #predictmodels.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.nn_shorttermforecast_fff")
  spark.sql("""delete from stats.nn_shorttermforecast_fff where fdate=to_date('{fdate}')""".format(fdate=fdate))
  predictmodels.write.format("delta").mode("append").saveAsTable("stats.nn_shorttermforecast_fff")
  print(spark.sql("""select * from stats.nn_shorttermforecast_fff where fdate=to_date('{fdate}')""".format(fdate=fdate)).count())
  print(predictmodels.count())
  print(predictmodels2.count())
  print(predictmodels3.count())
def toappendfinaltable():
  col=['process_date','fdate','saleyear','salemonth','ed_style_id','age','year','make','model','ed_model_id','tmv_category','msr_price','ptgfinal','dc','cc','dtt','flag']
  temp=spark.sql("""select * from stats.nn_shorttermforecast_fff where fdate=to_date('{fdate}')""".format(fdate=fdate)) \
     .withColumn("process_date",F.lit(processday)).select(*col)
  #temp.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("stats.nn_shorttermforecast_hist")
  spark.sql("""delete from stats.nn_shorttermforecast_hist where process_date=to_date('{processday}')""".format(processday=processday))
  temp.write.format("delta").mode("append").saveAsTable("stats.nn_shorttermforecast_hist")

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------

default_days='5'
default_modelyear='2018'
default_datapullsyyymm='201701'
default_filepath="/mnt/edmunds-rs/styles/tmvn"
default_epoch='35'
dbutils.widgets.text("days",default_days , " 1)forecast after days  ")
dbutils.widgets.text("syear",default_modelyear, " 2) vehicle start model year ")
dbutils.widgets.text("transminyymm",default_datapullsyyymm, " 3)transdata start month ")
dbutils.widgets.text("filepath", default_filepath, " 4)filepath")
dbutils.widgets.text("epoch", default_epoch, " 5) NN epoch")

# COMMAND ----------

days = int(dbutils.widgets.get('days') if dbutils.widgets.get('days') else default_days)
syear = int(dbutils.widgets.get('syear') if dbutils.widgets.get('syear') else default_modelyear)
transminyymm = int(dbutils.widgets.get('transminyymm') if dbutils.widgets.get('transminyymm') else default_datapullsyyymm)
filepath= dbutils.widgets.get('filepath') if dbutils.widgets.get('filepath') else default_filepath
epoch= int(dbutils.widgets.get('epoch') if dbutils.widgets.get('epoch') else default_epoch)

fdate=(date.today()+timedelta(days=days)).replace(day=1)
transmaxyymm=240001 #always forecast future
cyymm=int(fdate.strftime("%Y%m"))
processday=date.today()
print(processday)
print("forecasting for",cyymm,'after', days,'days from today',processday)

# COMMAND ----------

import mlflow
mlflow.set_tracking_uri('databricks')
mlflow.set_experiment("/Shared/experiments/nn_newvehicle_price")
    

# COMMAND ----------

mlflow.start_run()

# COMMAND ----------

start_datetime = datetime.now()
toAssignStartMM()
rawdata=toGetRawData(transmaxyymm,transminyymm,syear) ##forecast same month from May 18,2020
df1=toProcessTransData()
testdatap=toProcessTestData()
sumdata=toSummaryData()
vehiclelistf=toGetTestData()

# COMMAND ----------

nnlayers=['ed4','md','mk']
for nnlayer in nnlayers:
  datafset=toGetNNData(nnlayer)
  datasetf=NNmodel(epoch)
  #showValuationError()
  #displayTurningData()
  saveTodelta(nnlayer)
toCreateForecastdata()
toappendfinaltable()
end_datetime = datetime.now()

# COMMAND ----------

showValuationError()

# COMMAND ----------

truntime=((end_datetime-start_datetime).seconds//60)%60
traindatacnts=df1.count()
forecastedvehicles=spark.sql("""select * from stats.nn_shorttermforecast_hist where process_date=to_date('{processday}')""".format(processday=processday))
forecastvehiclecnts=forecastedvehicles.count()
pricefromlonterm=forecastedvehicles.filter('flag=99').count()
cnts_byyear=forecastedvehicles.groupBy('year').agg(F.count(F.col('ptgfinal')).alias("cnts")).toPandas()
cnts_bymake=forecastedvehicles.groupBy('make').agg(F.count(F.col('ptgfinal')).alias("cnts")).toPandas()


# COMMAND ----------


mlflow.log_param('processdate', processday)
mlflow.log_param('forecast days later', days)
mlflow.log_param('forecast month', cyymm)
mlflow.log_param('NN epoch', epoch)
mlflow.log_param('datapull start', transminyymm)
mlflow.log_param('oldest modelyear', syear)
mlflow.log_metric('totalruntime',truntime )
mlflow.log_metric('traindata_cnt', traindatacnts)
mlflow.log_metric('forecastedvehiclecnts', forecastvehiclecnts)
mlflow.log_metric('pricefromlongterm_ehicle_cnts', pricefromlonterm)


# COMMAND ----------

cnts_byyear.to_csv("cnts_byyear.csv",index=False,header=True)
cnts_bymake.to_csv("cnts_bymake.csv",index=False,header=True)
forecastedvehicles.toPandas().to_csv("newtmvprice.csv",index=False,header=True)
datasetf[0].astype(int).to_csv("trainevalutionerror.csv",index=False,header=True)
mlflow.log_artifact("cnts_byyear.csv")
mlflow.log_artifact("cnts_bymake.csv")
mlflow.log_artifact("newtmvprice.csv")
mlflow.log_artifact("trainevalutionerror.csv")
print(mlflow.get_artifact_uri())

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct ed_style_id) from stats.nn_shorttermforecast_fff 