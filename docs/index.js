URLS=[
"pyucalgarysrs/index.html",
"pyucalgarysrs/data/index.html",
"pyucalgarysrs/exceptions.html",
"pyucalgarysrs/models/index.html",
"pyucalgarysrs/pyucalgarysrs.html"
];
INDEX=[
{
"ref":"pyucalgarysrs",
"url":0,
"doc":""
},
{
"ref":"pyucalgarysrs.data",
"url":1,
"doc":""
},
{
"ref":"pyucalgarysrs.data.DataManager",
"url":1,
"doc":""
},
{
"ref":"pyucalgarysrs.data.DataManager.list_datasets",
"url":1,
"doc":"List available datasets",
"func":1
},
{
"ref":"pyucalgarysrs.data.DataManager.list_supported_read_datasets",
"url":1,
"doc":"List the datasets which have file reading capabilities supported.",
"func":1
},
{
"ref":"pyucalgarysrs.data.DataManager.check_if_read_supported",
"url":1,
"doc":"Check if a given dataset has file reading support",
"func":1
},
{
"ref":"pyucalgarysrs.data.DataManager.download",
"url":1,
"doc":"Download data from UCalgary open data archive",
"func":1
},
{
"ref":"pyucalgarysrs.data.DataManager.get_urls",
"url":1,
"doc":"Get URLs of data files",
"func":1
},
{
"ref":"pyucalgarysrs.data.DataManager.download_using_urls",
"url":1,
"doc":"Download data from UCalgary open data archive using a FileListingResponse object. This would be used in cases where more customization is needed than the generic  download function. For example, use  get_urls to retrieve the list of URLs available for download, further process this list to fewer files based on some other requirement (ie. time downsampling such as one file per hour), and then use this function to download the custom set URLs.",
"func":1
},
{
"ref":"pyucalgarysrs.data.DataManager.read",
"url":1,
"doc":"Read data files",
"func":1
},
{
"ref":"pyucalgarysrs.exceptions",
"url":2,
"doc":"The exceptions module contains exceptions unique to the PyAuroraX library"
},
{
"ref":"pyucalgarysrs.exceptions.SRSError",
"url":2,
"doc":"Common base class for all non-exit exceptions."
},
{
"ref":"pyucalgarysrs.exceptions.SRSInitializationError",
"url":2,
"doc":"Error during library initialization occurred"
},
{
"ref":"pyucalgarysrs.exceptions.SRSPurgeError",
"url":2,
"doc":"Error during purging of download or tar extraction working directory occurred"
},
{
"ref":"pyucalgarysrs.exceptions.SRSAPIError",
"url":2,
"doc":"Error during API call occurred"
},
{
"ref":"pyucalgarysrs.exceptions.SRSUnsupportedReadError",
"url":2,
"doc":"Unsupported dataset for read function"
},
{
"ref":"pyucalgarysrs.exceptions.SRSDownloadError",
"url":2,
"doc":"Unsupported dataset for read function"
},
{
"ref":"pyucalgarysrs.models",
"url":3,
"doc":""
},
{
"ref":"pyucalgarysrs.models.ModelsManager",
"url":3,
"doc":""
},
{
"ref":"pyucalgarysrs.pyucalgarysrs",
"url":4,
"doc":""
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS",
"url":4,
"doc":"Top-level class for interacting with UCalgary SRS data tools"
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS.DEFAULT_API_BASE_URL",
"url":4,
"doc":""
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS.DEFAULT_API_TIMEOUT",
"url":4,
"doc":""
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS.DEFAULT_API_HEADERS",
"url":4,
"doc":""
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS.download_output_root_path",
"url":4,
"doc":""
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS.read_tar_temp_dir",
"url":4,
"doc":""
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS.purge_download_output_root_path",
"url":4,
"doc":"",
"func":1
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS.purge_read_tar_temp_dir",
"url":4,
"doc":"",
"func":1
},
{
"ref":"pyucalgarysrs.pyucalgarysrs.PyUCalgarySRS.initialize_paths",
"url":4,
"doc":"",
"func":1
}
]