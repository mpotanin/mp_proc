import os
import argparse
import sys
import json

import zoning




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
                                     ('Collect stat by feature for all tile scenes'))

    parser.add_argument('-i', required=True, metavar='L2A folder',
                        help='Folder with sentinel 2 L2 products')
    parser.add_argument('-v', required=True, metavar='vector file',
                        help='Input vector file')
    parser.add_argument('-o', required=True, metavar='output file',
                        help='Output json file')
    parser.add_argument('-t', required=True, metavar='tile',
                        help='Tile name')
    parser.add_argument('-sd', required=False, metavar='start date yyyymmdd',
                        help='Start date filter yyyymmdd', default='20000000')
    parser.add_argument('-ed', required=False, metavar='end date yyyymmdd',
                        help='End date filter yyyymmdd', default='30000000')

    if (len(sys.argv) == 1):
        parser.print_usage()
        exit(0)
    args = parser.parse_args()



    all_stat = zoning.collect_tile_stat(args.v,args.i,args.t,args.sd,args.ed)

    with open(args.o, 'w') as fp:
        json.dump(all_stat, fp)


    print('OK!')
