#!/usr/bin/perl

#
# Initialization
#

use strict;
use warnings;

use List::Util qw(max min sum);
use List::MoreUtils;


#
# Routines
#

sub read_header {
        my $filename = shift;

        open(my $read, '<', $filename);
        chomp(my $header = <$read>);
        return unless ($header =~ s/^%\s+//);
        my @combinations = split(/\s+/, $header);
        close($read);
        return @combinations;
}

sub read_data {
        my $filename = shift;
        my @combinations = @_;
        @combinations = read_header($filename) unless @combinations;
        die("Column headers should be specified or present in the file itself.") unless @combinations;

        my %data;
        $data{$_} = [] foreach @combinations;

        open(my $read, '<', $filename);
        while (<$read>) {
                s/^\s*(.*\S)\s*$/$1/;
                next if m/^%/;
                my @array = split(/\s+/);
                map { push($data{$combinations[$_]}, $array[$_])} 0 .. $#array
        }

        return %data;
}

sub permute {
        my $outers = shift;
        my @inners = @_;
        return @{$outers} unless @inners;

        my @permutations;
        foreach my $outer (@{$outers}) {
                foreach my $inner (permute(@inners)) {
                        push @permutations, "$outer-$inner";
                }
        }
        return @permutations;
}

sub compare {
        my $datafile = shift;

        my @combinations = read_header($datafile)
                || die("Invalid datafile: could not read header.\n");

        my %data = read_data($datafile);
        my %reference = (
                read_data("reference-regular.dat", permute(['T1'..'T5'], ['P1'..'P3'])),
                read_data("reference-hermite.dat", permute(['T1'..'T5'], ['H1'..'H3']))
        );

        my $nrmse_total = 0;
        foreach my $combination (keys %data) {
                die("Not all data found in reference.")
                                unless defined $reference{$combination};
                die("Data trace length not equal to reference trace")
                        unless $#{$data{$combination}} == $#{$reference{$combination}};

                my $mse = 0;
                for my $i (0..$#{$data{$combination}}) {
                        my $diff = $data{$combination}->[$i] - $reference{$combination}->[$i];
                        $mse += $diff*$diff;
                }
                my $rmse = sqrt($mse / @{$data{$combination}});
                my $nrmse = $rmse / (
                        (max @{$data{$combination}}) - (min @{$data{$combination}})
                );

                printf "%s: %.2f%%\n", $combination, 100*$nrmse;
                $nrmse_total += $nrmse;
        }
        $nrmse_total /= keys %data;
        printf "       %.2f%%\n", 100*$nrmse_total;
}


#
# Main
#

my $datafile = shift || die("Provide datafile.");

my @combinations = read_header($datafile)
        || die("Invalid datafile: could not read header.\n");

my %data = read_data($datafile);
my %reference = (
        read_data("reference-regular.dat", permute(['T1'..'T5'], ['P1'..'P3'])),
        read_data("reference-hermite.dat", permute(['T1'..'T5'], ['H1'..'H3']))
);

my $nrmse_total = 0;
foreach my $combination (keys %data) {
        die("Not all data found in reference.")
                        unless defined $reference{$combination};
        die("Data trace length not equal to reference trace")
                unless $#{$data{$combination}} == $#{$reference{$combination}};

        my $mse = 0;
        for my $i (0..$#{$data{$combination}}) {
                my $diff = $data{$combination}->[$i] - $reference{$combination}->[$i];
                $mse += $diff*$diff;
        }
        my $rmse = sqrt($mse / @{$data{$combination}});
        my $nrmse = $rmse / (
                (max @{$data{$combination}}) - (min @{$data{$combination}})
        );

        printf "%s: %.2f%%\n", $combination, 100*$nrmse;
        $nrmse_total += $nrmse;
}
$nrmse_total /= keys %data;
printf "       %.2f%%\n", 100*$nrmse_total;
