Morphosyntactic Annotation for Urdu


This directory contains a duplicate of the Urdu corpus contained in the main directory 
structure which has been automatically tagged for part-of-speech.

It also contains the tagger software itself, and the source code for that software.

The full tagging project is described at length in Hardie (2004).


The tagset

The tagset used to tag the data is the U2 tagset. This is a reduced version of the U1 
tagset described in the accompanying document ("U1tagset.***"). While the U1 
tagset was designed for manual tagging, the U2 tagset was adjusted to optimise the 
tagset for an automatic tagger.

The differences between U1 and U2 are as follows:

*	U1 contains separate tags for proper and common nouns. U2 does not.
*	U1 contains a special tag, ZZ, for the izafat. U2 does not.

The Urdu tagset was created by Andrew Hardie, based on the grammatical description 
of Schmidt (1999).


The tagged data

All tagged files have filenames identical to the corresponding files in the main corpus, 
with a "-tgd" appended to indicate that it is a tagged version. The headers of the files, 
however, have not been changed.

The tagged files are organised into three directories – "parallel", "spoken" and 
"written" – according to their provenance in the original corpus.

In the tagged files, words and their tags are given in the following format:

<w pos="list of possible tags">word</w>


The tagger software

The tagger software has been compiled for use on 32 bit Windows systems. However 
the source code is also given (see below) allowing the programs to be recompiled for 
other systems.

The tagger software is an instantiation of the Unitag Unicode-compliant part-of-
speech tagger. It uses a custom-made analyser (Urdutag) to supply possible analyses 
to words. It then removes contextually inappropriate analyses using the Unirule 
program. Unirule is a rule-based disambiguator which applies hand-written rules to 
reduce ambiguity.

There are eight programs supplied in the "Software" directory. The only one 
described in full here is urduwrap, which makes a direct call to the Unitag system. 
This makes the process of tagging a file very easy for the user, as urduwrap sets all 
the options for you. For full details on Unitag, Unirule, Unilex and the associated 
programs, see Hardie (2004).

The accuracy of the tagger is circa 90% with a very high ambiguity level. This is 
rather poor by the standards of many contemporary taggers for languages such as 
English, but is at the moment the best that can be achieved due to the limitations 
imposed by the very small lexicon. This lexicon ("urdulexicon.txt") only contains 
around 8,000 word-types.

Using Urduwrap

To use Urduwrap to call the tagging system, put all the programs and text files in the 
"software" directory into the same directory as the files you wish to tag.

Then, from an MS-DOS command prompt, give the following command:

urduwrap list.txt

where list.txt (or whatever other filename you choose to use) is an ASCII text file 
containing a list of all the files to be tagged, one filename per line, and nothing else.

All files to be tagged must be two-byte Unicode text.


The source code

The tagger was written in the C programming language.

The source code is included in the "Code" directory.

Many of the files are common to several programs. To work out which files are 
needed for which program, do the following:

*	Identify the main( ) function. The main( ) function of each program is 
contained in the *.c file which has the same name as that program, or in a 
separate file whose filename indicates that it contains a main function (e.g. 
"urtgmain.c" contains the main( ) function for the program Urdutag).

*	Identify the other source files needed. This can be done by reference to the 
header files #included in each program. For instance, if a file #includes the 
header "unicode.h", then that program requires functions in the file 
"unicode.c". This is iterative, e.g. "commandline.c" #includes "andrew.h" and 
thus any program that uses functions in "commandline.c" also requires 
"andrew.c".

Most of the code is ANSI standard C and should therefore compile on any system. 
However, some systems may have trouble compiling the general-function file 
andrew.c due to a non-standard library header file being used.

If this causes you difficulties, in the file "andrew.c" remove the following line:

#include <conio.h>

and replace it with the following lines:

#define getch XXX
#define getche XXX

where XXX is the name of a keyboard character input function that is unbuffered and 
takes no arguments. This will allow the programs to compile (although there may be 
bugs).


References

Hardie, A (2004) The computational analysis of morphosyntactic categories in Urdu. 
PhD Thesis, Lancaster University.

Schmidt, RL (1999) Urdu: an essential grammar. London: Routledge.
