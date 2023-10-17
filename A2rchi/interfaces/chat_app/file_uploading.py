#TODO: SEE HOW MUCH OF THE BELOW WE NEED
from A2rchi.utils.config_loader import Config_Loader
from A2rchi.utils.env import read_secret
from A2rchi.utils.scraper import Scraper

from flask import render_template, request, redirect, url_for, flash, session

import hashlib
import os
import yaml

def simple_hash(input_string):
    """
    Takes an input string and outputs a hash
    """

    # perform the hash operation using hashlib
    identifier = hashlib.md5()
    identifier.update(input_string.encode('utf-8'))
    hash_value= str(int(identifier.hexdigest(), 16))

    return hash_value


def file_hash(filename):
    """
    Takes an input filename and converts it to a hash.
    
    However, unlike `simple_hash` this method keeps the file extension
    at the end of the name
    """

    return simple_hash(filename)[0:12] + "." + filename.split(".")[-1]


def add_filename_to_filehashes(filename, data_path, filehashes_yaml_file="manual_file_hashes.yaml"):
    """
    Adds a filename and its respective hash to the map between filenames and hashes

    Map is stored as a .yml file in the same path as where the data is stored. Keys are the hashes 
    and values are the filenames

    Returns true if hash was able to be added sucsessfully. Returns false if the hash (and thus likely)
    the filename already exists.
    """
    hash_string = file_hash(filename)
    try:
        # load existing hashes or initialize as empty dictionary
        with open(os.path.join(data_path, filehashes_yaml_file), 'r') as file:
            filenames_dict = yaml.safe_load(file) or {}
    except FileNotFoundError:
        filenames_dict = {}

    # check if the file already exists
    if hash_string in filenames_dict.keys():
        print(f"File '{filename}' already exists.")
        return False

    # add the new filename and hashed file string to the accounts dictionary
    filenames_dict[hash_string] = filename

    # write the updated dictionary back to the YAML file
    with open(os.path.join(data_path, filehashes_yaml_file), 'w') as file:
        yaml.dump(filenames_dict, file)

    return True


def remove_filename_from_filehashes(filename, data_path, filehashes_yaml_file="manual_file_hashes.yaml"):
    """
    Removes a filename and its respective hash from the map between filenames and hashes

    Map is stored as a .yml file in the same path as where the data is stored. Keys are the hashes 
    and values are the filenames

    Always returns true
    """
    hash_string = file_hash(filename)
    try:
        # load existing accounts or initialize as empty dictionary
        with open(os.path.join(data_path, filehashes_yaml_file), 'r') as file:
            filenames_dict = yaml.safe_load(file) or {}
    except FileNotFoundError:
        filenames_dict = {}

    # check if the filename already exists and remove if it does
    if hash_string in filenames_dict.keys():
        filenames_dict.pop(hash_string)

    # write the updated dictionary back to the YAML file
    with open(os.path.join(data_path, filehashes_yaml_file), 'w') as file:
        yaml.dump(filenames_dict, file)

    return True


def get_filename_from_hash(hash_string, data_path, filehashes_yaml_file="manual_file_hashes.yaml"):
    """
    Given a file hash, returns the original file name from the map chat_app
    """
    try:
        # load existing accounts or initialize as empty dictionary
        with open(os.path.join(data_path, filehashes_yaml_file), 'r') as file:
            filenames_dict = yaml.safe_load(file) or {}
    except FileNotFoundError:
        filenames_dict = {}

    return filenames_dict[hash_string] if hash_string in filenames_dict else None


def remove_url_from_sources(url, sources_path):
    try:
        # load existing accounts or initialize as empty dictionary
        with open(sources_path, 'r') as file:
            sources = yaml.safe_load(file) or {}
    except FileNotFoundError:
        sources = {}

    # check if the url already exists and remove if it does
    sources = {k:v for k,v in sources.items() if v != url}

    # write the updated dictionary back to the YAML file
    with open(sources_path, 'w') as file:
        yaml.dump(sources, file)