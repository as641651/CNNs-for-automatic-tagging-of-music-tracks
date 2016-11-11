#!/usr/bin/env python
"""
Runs one boundary algorithm and a label algorithm on a specified audio file
and outputs the results using the MIREX format.
"""
"""This script is taken from Music structural Analysis Framework (urinieto/MSAF)
   and modified by Aravind Sankaran, RWTH"""

import argparse
import logging
import time
import utils
import json
# MSAF import
import msaf

"""Runs the segmentation algorithm on the music tracks.
   
   This script depends on the Music-Structural-Analysis-Framework:
   https://github.com/urinieto/msaf"""

OKGREEN = '\033[92m'
ENDC = '\033[0m'

def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(
        description="Runs the speficied algorithm(s) on the input file and "
        "the results using the MIREX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-bid",
                        action="store",
                        help="Boundary algorithm identifier",
                        dest="boundaries_id",
                        default=msaf.config.default_bound_id,
                        choices=["gt"] +
                        msaf.io.get_all_boundary_algorithms())
    parser.add_argument("-lid",
                        action="store",
                        help="Label algorithm identifier",
                        dest="labels_id",
                        default=msaf.config.default_label_id,
                        choices=msaf.io.get_all_label_algorithms())
    parser.add_argument("-d",
                        action="store",
                        dest="dirPath",
                        default= "/home/sankaran/Thesis/data/Waves",
                        help="Path to Main folder")
    parser.add_argument("-f",
                        action="store",
                        dest="folder",
                        default= "",
                        help="Path to a sub folder")
    parser.add_argument("-o",
                        action="store",
                        dest="out_file",
                        help="Output json file with the results")

    args = parser.parse_args()
    start_time = time.time()

    if not args.out_file:
       print "Please Specify path for outfile"
       exit()
    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run MSAF
    params = {
        "annot_beats": False,
        "feature": "cqt",
        "framesync": False,
        "boundaries_id": args.boundaries_id,
        "labels_id": args.labels_id,
        "n_jobs": 1,
        "hier": False,
        "sonify_bounds": False,
        "plot": False
    }
  
    database = {}
    count = 0 
    tracks = utils.get_tracks(data_folder=args.dirPath, folder = args.folder)
    total = len(tracks)
    for t in tracks:
       print OKGREEN + "%d/%d"%(count,total) + ENDC + t
       database[count] = {}
       database[count]['name'] = t.split("/")[-1]
       database[count]['year'] = t.split("/")[-2]
       res = msaf.run.process(t, **params)
       database[count]['segments'] = res[0].tolist()
       with open(args.out_file,"w") as f:
          f.write(json.dumps(database,sort_keys=True,indent=4))
       count = count + 1
       #msaf.io.write_mirex(res[0], res[1], args.out_file)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()

