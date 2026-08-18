#!/usr/bin/env perl
use strict;
use warnings;
use JSON::PP qw(decode_json encode_json);

my ($input, $output) = @ARGV;
die "usage: standardize_queryrewrite_think.pl INPUT OUTPUT\n" unless defined $output;

open my $in,  '<', $input  or die "cannot read $input: $!\n";
open my $out, '>', $output or die "cannot write $output: $!\n";

my $replacement = '<think>The retrieved evidence now supports the answer.</think>';
my ($rows, $changed) = (0, 0);

while (my $line = <$in>) {
    chomp $line;
    next if $line eq '';
    my $record = decode_json($line);
    $rows++;

    if (($record->{sample_type} // '') eq 'final_answer'
        && ($record->{trajectory_type} // '') eq 'single_search_hybrid_v1_queryrewrite_real_retrieval') {
        my $old = $record->{target_text} // '';
        my $new = $old;
        $new =~ s{<think>.*?</think>}{$replacement}s;
        die "missing target think block for $record->{original_id}\n" if $new eq $old;
        $record->{target_text} = $new;
        $changed++;
    }

    print {$out} encode_json($record), "\n";
}

close $in;
close $out;
die "expected 1850 QueryRewrite replacements, got $changed\n" if $changed != 1850;
print "Processed $rows rows; standardized $changed QueryRewrite target think blocks.\n";
