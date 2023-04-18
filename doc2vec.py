import gensim
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import string
import os

import SwiftUtils

from datetime import datetime

import pickle

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time
# MongoClass download & creation
if not os.path.exists('MongoClass.py'):
    SwiftUtils.download_object('UC2_Similar_Customer', 'MongoClass.py', '.')

logger.info('MongoClass.py downloaded or present')
    
from MongoClass import MongoDataAccess

db_config= {
    "MONGODB_ADDRESS"    : os.environ.get('KEY_MONGO_ADDRESS',''),
    "MONGODB_PORT"       : os.environ.get('KEY_MONGO_PORT', ''),
    "MONGODB_USERNAME"   : os.environ.get('KEY_MONGO_USER', ''),
    "MONGODB_PASSWORD"   : os.environ.get('KEY_MONGO_PASSWORD', ''),
    "MONGODB_DATABASE"   : os.environ.get('KEY_MONGO_DATABASE', '')
}

dataAccessObj = MongoDataAccess(db_config,'IBSO')

# Class definition
class Doc2VecModel:
    def __init__(self):
        # If model file(s) aren't in filesystem, download them
        container = "UC2_MR_Similar_Customer"
        model_path = 'Doc2Vec_v1.model'
        model_aux_path = 'Doc2Vec_v1.model.dv.vectors.npy'
        if not (os.path.exists(model_path) and os.path.exists(model_aux_path)):
            SwiftUtils.download_object(container, model_path, '.')
            SwiftUtils.download_object(container, model_aux_path, './')
        
        logger.info('model files downloaded or present')
        
        # Load the doc2vec model
        self.model = Doc2Vec.load(model_path)
        
        logger.info('model loaded')

        # Info to get embeddings later
        self.embeddings_coll_name = "UC2_Doc2Vec_Embeddings_v2"
        self.relevant_field = 'description'
        self.embeddings_field = self.relevant_field + '_embeddings'
        self.maximum_records = 5000

        # datetime_filter = {
        #     'incident_created_at_dt': {}
        # }
        # projection = {
        #     'css_object_id': True
        # }
        # projection[self.embeddings_field] = True
        # embeddings_coll = dataAccessObj.connect_mongo()['IBSO'][self.embeddings_coll_name]
        # if not os.path.exists('/data/css_object_id_list.pkl'):
        #     css_object_id_list = []
        #     for i in [2018, 2019, 2020, 2021]:
        #         datetime_filter['incident_created_at_dt']['$gte'] = datetime(i, 1, 1)
        #         datetime_filter['incident_created_at_dt']['$lt'] = datetime(i, 6, 1)
        #         np_embeddings = []
        #         cursor = embeddings_coll.find(datetime_filter, projection)
        #         for record in cursor:
        #             css_object_id_list.append(record['css_object_id'])
        #             np_embeddings.append(record[self.embeddings_field])
        #         np_embeddings = np.array(np_embeddings).astype('float16')
        #         with open('/data/embeddings_' + str(i) + '_1.pkl', 'wb') as output_file:
        #             pickle.dump(np_embeddings, output_file)
        #         del np_embeddings

        #         datetime_filter['incident_created_at_dt']['$gte'] = datetime(i, 6, 1)
        #         datetime_filter['incident_created_at_dt']['$lt'] = datetime(i + 1, 1, 1)
        #         np_embeddings = []
        #         cursor = embeddings_coll.find(datetime_filter, projection)
        #         for record in cursor:
        #             css_object_id_list.append(record['css_object_id'])
        #             np_embeddings.append(record[self.embeddings_field])
        #         np_embeddings = np.array(np_embeddings).astype('float16')
        #         with open('/data/embeddings_' + str(i) + '_2.pkl', 'wb') as output_file:
        #             pickle.dump(np_embeddings, output_file)
        #         del np_embeddings
        #         logger.info("Downloaded embeddings for " + str(i))
        #     with open('/data/css_object_id_list.pkl', 'wb') as output_file:
        #         pickle.dump(css_object_id_list, output_file)
        
    # Input: css_object_id - css_object_id of infodoc we want to get similar to
    # Input: n - number of results we want back
    # Output: DataFrame with cols: css_object_id, description, similarity and n rows
    def compute_top_n(self, css_object_id, n, relevant_field, maximum_records, filter_by_prod_version, filter_by_component):
        self.relevant_field = relevant_field
        self.embeddings_field = self.relevant_field + '_embeddings'
        self.maximum_records = maximum_records
        metadata = self.get_metadata_helper(css_object_id)
        input_embeddings = np.reshape(self.create_embedding_helper(metadata[self.relevant_field]), (1, -1))

        t0 = time.time()
        embeddings_coll = dataAccessObj.connect_mongo()['IBSO'][self.embeddings_coll_name]
        query = {}
        projection = {
            'css_object_id': True,
            self.embeddings_field: True
        }
        if filter_by_component:
            if metadata["component"] and len(metadata["component"]) > 0:
                query["component"] = metadata["component"]
        if filter_by_prod_version:
            if metadata["prod_version"] and len(metadata["prod_version"]) > 0:
                query["prod_version"] = metadata["prod_version"]
        query['incident_created_at_dt'] = {
            '$lt': metadata['incident_created_at_dt']
        }
        embeddings_df = pd.DataFrame(list(embeddings_coll.find(query, projection).sort('incident_created_at_dt', -1).limit(self.maximum_records)))
        np_embeddings = np.array(embeddings_df[self.embeddings_field].tolist())
        cosine_results = cosine_similarity(input_embeddings, np_embeddings)

        df = pd.DataFrame(columns=['css_object_id', self.relevant_field, 'similarity'])
        for i in np.argsort(cosine_results[0])[::-1][0:n]:
            matched_css_object_id = embeddings_df.iloc[i]['css_object_id']
            matched_field = self.get_relevant_field_helper(matched_css_object_id)
            df = df.append({
                'css_object_id': matched_css_object_id,
                self.relevant_field: matched_field,
                'similarity': cosine_results[0][i]
            }, ignore_index=True)
        logger.info(time.time() - t0)
        return df

        # description = self.get_relevant_field_helper(css_object_id)
        # return self.compute_top_n_freetext(description, n)
        
    # Input: description - unprocessed description we want to get similar to
    # Input: n - number of results we want back
    # Output: DataFrame with cols: css_object_id, description, similarity and n rows
    def compute_top_n_freetext(self, description, n):
        # Get embeddings for input description
        input_embeddings = np.reshape(self.create_embedding_helper(description), (1, -1))
        
        # Get cosine similarities between input embeddings and all saved embeddings
        t0 = time.time()
        with open('/data/css_object_id_list.pkl', 'rb') as input_file:
            css_object_id_list = pickle.load(input_file)
        cosine_results = np.array([[]])
        
        for i in [2018, 2019, 2020, 2021]:
            t1 = time.time()
            with open('/data/embeddings_' + str(i) + '_1.pkl', 'rb') as input_file:
                np_embeddings = pickle.load(input_file)
            t2 = time.time()
            logger.info("time to open 1, " + str(i) + ": " + str(t2 - t1))
            tmp_sim = cosine_similarity(input_embeddings, np_embeddings)
            t3 = time.time()
            logger.info("time to get similarity:" + str(t3 - t2))
            cosine_results = np.concatenate((cosine_results, tmp_sim), axis=1)
            t4 = time.time()
            logger.info("time to concat to results:" + str(t4 - t3))
            del tmp_sim
            del np_embeddings
            with open('/data/embeddings_' + str(i) + '_2.pkl', 'rb') as input_file:
                np_embeddings = pickle.load(input_file)
            t5 = time.time()
            logger.info("time to open 2, " + str(i) + ": " + str(t5 - t4))
            tmp_sim = cosine_similarity(input_embeddings, np_embeddings)
            t6 = time.time()
            logger.info("time to get similarity:" + str(t6 - t5))
            cosine_results = np.concatenate((cosine_results, tmp_sim), axis=1)
            logger.info("time to concat to results:" + str(time.time() - t6))
            del tmp_sim
            del np_embeddings

        logger.info(time.time() - t0)

        # Create empty DataFrame to populate and return
        df = pd.DataFrame(columns=['css_object_id', 'description', 'similarity'])
        
        # Sort similarities descending, grab first n, then populate a row in the return DataFrame for each
        for i in np.argsort(cosine_results[0])[::-1][0:n]:
            matched_css_object_id = css_object_id_list[i]
            matched_description = self.get_relevant_field_helper(matched_css_object_id)
            df = df.append({
                'css_object_id': matched_css_object_id,
                'description': matched_description,
                'similarity': cosine_results[0][i]
            }, ignore_index=True)
        return df
        
    # Input: css_object_id - css_object_id of infodoc we want to compare against
    # Input: css_object_ids - list of all other css_object_ids (already in embeddings) we want to compare first input against
    # Output: DataFrame with cols: css_object_id, relevant field's name, similarity and len(css_object_ids) rows
    def compute_similarity_specified(self, css_object_id, css_object_ids, relevant_field):
        self.relevant_field = relevant_field
        self.embeddings_field = self.relevant_field + '_embeddings'
        field_value = self.get_relevant_field_helper(css_object_id)
        return self.compute_similarity_specified_freetext(field_value, css_object_ids)
        
    # Input: input_value - unprocessed input we want to compare against
    # Input: css_object_ids - list of all other css_object_ids (already in embeddings) we want to compare first input against
    # Output: DataFrame with cols: css_object_id, relevant field's name, similarity and len(css_object_ids) rows
    def compute_similarity_specified_freetext(self, input_value, css_object_ids):
        # Get embeddings for input
        input_embeddings = self.create_embedding_helper(input_value)

        mongo_query = {
            'css_object_id': {
                '$in': css_object_ids
            }
        }

        mongo_projection = {
            'css_object_id': True
        }
        mongo_projection[self.embeddings_field] = True
        
        # Get all embeddings with css_object_id matching something in css_object_ids
        filtered_embeddings = pd.DataFrame(list(dataAccessObj.connect_mongo()['IBSO'][self.embeddings_coll_name].find(mongo_query, mongo_projection)))
        
        # Get cosine similarities between input embeddings and filtered saved embeddings
        cosine_results = cosine_similarity(np.reshape(input_embeddings,(1, -1)), filtered_embeddings[self.embeddings_field].values.tolist())
        
        # Create empty DataFrame to populate and return
        df = pd.DataFrame(columns=['css_object_id', self.relevant_field, 'similarity'])
        
        # Sort similarities descending, then populate a row in the return DataFrame for each
        for i in np.argsort(cosine_results[0])[::-1]:
            matched_css_object_id = filtered_embeddings.iloc[i]['css_object_id']
            matched_field = self.get_relevant_field_helper(matched_css_object_id)
            df = df.append({
                'css_object_id': matched_css_object_id,
                self.relevant_field: matched_field,
                'similarity': cosine_results[0][i]
            }, ignore_index=True)
        return df
        
    # All the following are never used outside the class, don't need to be there in your classes
    
    # Input: css_object_id - css_object_id of infodoc we want to get the description from
    # Output: string of description
    def get_relevant_field_helper(self, css_object_id):
        metadata_coll = 'infodocs_metadata'
        return dataAccessObj.connect_mongo()['IBSO'][metadata_coll].find_one({
            'css_object_id': css_object_id
        }, {
            self.relevant_field: True
        })[self.relevant_field]

    def get_metadata_helper(self, css_object_id):
        metadata_coll = 'infodocs_metadata'
        return dataAccessObj.connect_mongo()['IBSO'][metadata_coll].find_one({
            'css_object_id': css_object_id
        })
    
    # Input: input_value - unprocessed input to get embeddings for
    # Output: list of embeddings
    def create_embedding_helper(self, input_value):
        processed_input = self.textProcessing(input_value)
        tagged_doc = TaggedDocument(simple_preprocess(processed_input), None)
        return self.model.infer_vector(tagged_doc[0]).tolist()
    
    # Input: input_value - unprocessed input to process
    # Output: string processed input
    def textProcessing(self, input_value):
        input_value = input_value.lower()
        input_value = re.sub('[%s]' % re.escape(string.punctuation.replace('-', '')), ' ', input_value)
        input_value = re.sub('\d', '#', input_value)
        return input_value