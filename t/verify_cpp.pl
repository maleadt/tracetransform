#!/usr/bin/perl

#
# Initialization
#

use strict;
use warnings;


#
# Main
#

my $folder = shift || die("Provide build folder.\n");
system("make", "-C", $folder)
        == 0 || die("Could not compile transform.\n");;

system("$folder/transform", "--verbose", "-i", "Cam1_V1.pgm",
        (map { ("-T", $_ ) } 1..5),
        (map { ("-P", $_ ) } 1..3),
        "-o", "data-regular.dat")
        == 0 || die("Could not run transform.\n");
system("perl", "verify.pl", "data-regular.dat")
        == 0 || die("Could not verify.\n");

system("$folder/transform", "--verbose", "-i", "Cam1_V1.pgm",
        (map { ("-T", $_ ) } 1..5),
        (map { ("-P", $_ ) } 'H1'..'H3'),
        "-o", "data-hermite.dat")
        == 0 || die("Could not run transform.\n");
compare("perl", "verify.pl", "data-hermite.dat")
        == 0 || die("Could not verify.\n");
