#!/usr/bin/perl

# Write nicely
use strict;
use warnings;

# Required packages
use Getopt::Long;
use Cwd;

# Parse arguments
my $mode;
my (@tfunctionals, @pfunctionals);
my $iterations;
my $angle;
my $directory;
my $unsupported;
GetOptions(
    "quiet|q"           => \$unsupported,
    "verbose|v"         => \$unsupported,
    "debug|d"           => \$unsupported,
    "t-functional|T=i"  => \@tfunctionals,
    "p-functional|P=i"  => \@pfunctionals,
    "mode|m=s"          => \$mode,
    "iterations|i=i"    => \$iterations,
    "angle|a=i"         => \$angle,

    "directory=s"       => \$directory,
) or exit(1);

# Check argument validity
my $input = shift || die("No input specified");
warn("WARNING: unsupported arguments used\n") if ($unsupported);
die("ERROR: unknown argument(s) ", join(", ", @ARGV), "\n") if (@ARGV);
$directory = getcwd unless defined $directory;
$angle = 1 unless defined $angle;
die("ERROR: invalid program mode") unless $mode =~ m"^(benchmark|calculate)$";
die("ERROR: required argument iterations was not provided\n")
    if ($mode eq "benchmark" && not defined $iterations);
die("ERROR: required argument t-functional was not provided\n")
    unless @tfunctionals;
die("ERROR: input does not exist\n") unless (-f $input);

# Construct the MATLAB command
my $matlab_tfunctionals = "[" . join(" ", @tfunctionals) . "]";
my $matlab_pfunctionals = "[" . join(" ", @pfunctionals) . "]";
my $matlab_setup = "imageFile='$input';" .
               " tfunctionals=$matlab_tfunctionals;" .
               " pfunctionals=$matlab_pfunctionals;" .
               " program_mode='$mode';" .
               " directory='$directory'";
if ($iterations) {
    $matlab_setup .= "; iterations=$iterations";
}
if ($angle) {
    $matlab_setup .= "; angle_interval=$angle";
}
my $matlab_code = "try, run('$directory/demo.m'), catch err, fprintf(2, '%s\\n'," .
              " getReport(err, 'extended')); exit(1), end, exit(0)";
my $cmd = "matlab -nodisplay -r \"$matlab_setup; $matlab_code\"";

# Execute command and print output (but strip the MATLAB header)
open(my $output, "$cmd |");
my $line = 0;
while (<$output>) {
    print if (++$line > 10);
}
close($output);