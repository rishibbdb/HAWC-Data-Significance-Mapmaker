from __future__ import print_function
from builtins import range
from hawc_hal.maptree import map_tree_factory
from astropy.io import fits

from IPython import embed

import healpy as hp
import numpy as np

import argparse
import os
from datetime import datetime

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="HAWC Map Maker: Convert HDF5 to FITS and generate maps, or use existing FITS files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Convert residual HDF5 to FITS, then make maps
  %(prog)s hdf5 -d data/ -f residual_fit.hd5 -o output/ -p residual \
    --ra 45.0 --dec -15.0 --roi-x 5 --roi-y 5 \
    --det-res response.fits

  # Use existing FITS files with manual bin list
  %(prog)s fits -d data/ -b B7C0Ej B8C0Ej B9C0Ej \
    --ra 45.0 --dec -15.0 --roi-x 5 --roi-y 5 \
    --det-res response.fits
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)
    
    # Mode 1: HDF5 to FITS then maps
    hdf5_parser = subparsers.add_parser('hdf5', help='Convert HDF5 to FITS then make maps')
    
    hdf5_parser.add_argument(
        '-d', '--data-dir',
        type=str,
        required=True,
        help='Directory containing HDF5 file'
    )
    
    hdf5_parser.add_argument(
        '-f', '--filename',
        type=str,
        required=True,
        help='HDF5 filename (e.g., residual_fit.hd5)'
    )
    
    hdf5_parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Output directory for FITS files and maps'
    )
    
    hdf5_parser.add_argument(
        '-p', '--prefix',
        type=str,
        default='map',
        help='Prefix for output FITS files (default: map)'
    )
    
    hdf5_parser.add_argument(
        '-M', '--map-output',
        type=str,
        default='map.fits',
        help='Output map filename (default: map.fits)'
    )

    return parser


def convert_hd5_to_fits(dir, filename, outfile):
    input_filepath = os.path.join(dir, filename)
    if not os.path.isfile(input_filepath):
        print("You must enter a valid file!")
        exit(1)


    # Export the entire map tree (full sky)
    # this throws a warning for partial map trees but its OK
    maptree = map_tree_factory(input_filepath, None)


    now=datetime.now()
    startMJD=56987.9286332


    #FIRST HEADER
    '''
    COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
    COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H 
    DATE    = '2018-12-01T02:31:14' / file creation date (YYYY-MM-DDThh:mm:ss UT)   
    STARTMJD=     56987.9286332451 / MJD of first event                             
    STOPMJD =     58107.2396848326 / MJD of last event                              
    NEVENTS =                  -1. / Number of events in map                        
    TOTDUR  =     24412.9020670185 / Total integration time [hours]                 
    DURATION=      1.9943578604616 / Avg integration time [hours]                   
    MAPTYPE = 'duration'           / e.g. Skymap, Moonmap                           
    MAXDUR  =                  -1. / Max integration time [hours]                   
    MINDUR  =                  -1. / Min integration time [hours]                   
    EPOCH   = 'unknown '           / e.g. J2000, current, J2016, B1950, etc.        
    HIERARCH MAPFILETYPE = 'duration' / e.g. standard, duration, integration   
    '''

    FITS_COMMENT="FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"

    primary_keys=['COMMENT', 'COMMENT', 'DATE', 'STARTMJD', 'STOPMJD',
                'NEVENTS', 'TOTDUR', 'DURATION', 'MAPTYPE', 'MAXDUR', 
                'MINDUR', 'EPOCH', 'MAPFILETYPE']

    primary_values=[FITS_COMMENT  ,   FITS_COMMENT, "{0}".format(now), 56987.9286332451, 58107.2396848326,
                -1.0, 24412.9020670185, 1.9943578604616, 'duration', -1.0, -1.0, 'unknown', 'duration']

    primary_comments=["file does conform to FITS standard",
                    "number of bits per data pixel",
                    "number of data axes",
                    "FITS dataset may contain extension",
                    "MJD of first event", "MJD of last event",
                    "Number of events in map",
                    "Total integration time [hours]",
                    "Avg integration time [hours]",
                    "e.g. Skymap, Moonmap",
                    "Max integration time [hours]",
                    "Min integration time [hours]",
                    "e.g. J2000, current, J2016, B1950, etc.",
                    "e.g. standard, duration, integration"]



    labels=['data map', 'background map', 'exposure map']
    label_format=[ np.float64 for i in range(len(labels)) ]
    label_units=[ 'unknown' for i in range(len(labels)) ]


    for i, analysis_bin in enumerate(maptree.analysis_bins_labels):
        map_bin    = maptree[analysis_bin]
        #properties
        nside      = map_bin.nside
        npix       = map_bin.npix
        see_pixels = map_bin.observation_map._pixels_ids
        transits   = map_bin.n_transits
        scheme     = map_bin.scheme

        nest_scheme=False
        if scheme.lower()=='nested':
            nest_scheme=True

        #what we want
        data  = map_bin.observation_map.as_dense()
        bkg   = map_bin.background_map.as_dense()

        zeros = np.empty(npix)
        zeros.fill(9e9)

        outFileName=os.path.join(dir, "{0}_bin{1}.fits.gz".format(outfile,analysis_bin))
        #I probably need to add some header info, but yeah
        hp.fitsfunc.write_map(outFileName, (data, bkg, zeros), 
                            column_names=labels, column_units=label_units, dtype=label_format, 
                            partial=False, fits_IDL=True, overwrite=True, nest=nest_scheme)


        #add the cards to the header
        with fits.open(outFileName,'update') as hdu1:
            hdr=hdu1[0].header

            for i,key in enumerate(primary_keys):
                val=primary_values[i]
                comment=primary_comments[i]

                if key=='TOTDUR':
                    val=24.0*transits
            
                if key=='STOPMJD':
                    val=startMJD+transits

                entry=(val,comment)

                hdr[key]=entry

            #File is now closed
        print("File Written: {0}".format(outFileName))


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    convert_hd5_to_fits(dir=args.data_dir,filename=args.filename,outfile=args.prefix)

if __name__ == '__main__':
    exit(main())
