import pastebin
import time
import sys
from datetime import timedelta
import os.path

APIKEY = "@PASTEBIN_API_KEY@"
USERNAME = "@PASTEBIN_USERNAME@"
PASSWORD = "@PASTEBIN_PASSWORD@"
PRIVATEKEY = ""

pname = time.strftime("%Y%m%d-%H%M%S")

def GetUpTime():
    with open('/proc/uptime', 'r') as f:
        uptime_seconds = float(f.readline().split()[0])
        uptime_string = str(timedelta(seconds=uptime_seconds))
    return uptime_string

def GetAgentsBanner():
    str = ''.join([
        "##############################################\n",
        "  Python PasteBin Agent V1.0\n",
        "  Instance Up-time: ", GetUpTime(), '\n',
        "  Date: ", pname, '\n',
        "##############################################\n\n",
    ])
    return str

def ReadLogFile(fname):
    f = open(fname, "r")
    content = f.read()
    f.close()
    final_content = ''.join([GetAgentsBanner(), content])
    return final_content

def LoginToPasteBin():
    private_key = pastebin.generate_user_key(APIKEY, USERNAME, PASSWORD)
    return private_key

def TryUpload(fname, pastename="AutoBuild-", mode="HW-"):
    success = True
    try:
        PRIVATEKEY = LoginToPasteBin()
        logcontent = ReadLogFile(fname)
        joinedpname = ''.join([pastename, mode, pname])
        pastebin.paste(APIKEY, logcontent, api_user_key=PRIVATEKEY, paste_private='private', paste_name=joinedpname)
    except not pastebin.PastebinError:
        print("An exception occurred")
        success = False
    except pastebin.PastebinError as e:
        print "\tPasteBin Controller Handled Exception:"
        print ''.join(['\t', str(e)])
    return success

def MainFunc():
    print "Running python pastebin log uploader agent..."
    rslt1 = True
    rslt2 = True
    rslt3 = True

    fname = "autobuild_hw_log.txt"
    if os.path.isfile(fname):
        print "Found HW log file."
        rslt1 = TryUpload(fname, mode="HW-")

    fname = "autobuild_hwemu_log.txt"
    if os.path.isfile(fname):
        print "Found HW-EMU log file."
        rslt2 = TryUpload(fname, mode="HWEMU-")

    fname = "autobuild_swemu_log.txt"
    if os.path.isfile(fname):
        print "Found SW-EMU log file."
        rslt3 = TryUpload(fname, mode="SWEMU-")

    rslt = rslt1 and rslt2 and rslt3
    if rslt:
        print "Agent has done its work successfully."
        sys.exit(0)
    else:
        print "Agent has failed."
        sys.exit(5)

MainFunc()
