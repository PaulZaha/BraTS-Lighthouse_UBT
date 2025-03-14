import synapseclient
import os
from logging_config import logger
import pandas as pd

def init_data(training_cache: str, validation_cache: str):
    """
    Requires Data to be included into the download cart in Synapse.
    """
    syn = synapseclient.Synapse()
    syn.login(authToken=os.environ.get('synapse_token'))
    cache_dir = syn.cache.cache_root_dir
    print(f"Synapse Cache Directory: {cache_dir}")

    try:
        logger.info('data download started')
        
        dl_list_file_entities = syn.get_download_list()
        logger.info('successfully downloaded data')

    except synapseclient.client.exceptions.SynapseHTTPError:
        logger.error('no data in download cart')



def main():

    init_data()



if __name__ == '__main__':
    main()