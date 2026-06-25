#include <stdio.h>
#include "unicode.h"
#include "tagfile.h"
#include "lexicon.h"
#include "u1_tagset.h"
#include "urdutag.h"

/* urdutag input_filename output_filename lexicon_filename */
main(int argc, char *argv[])
{

	/* check args */

	if (argc != 4)
	{
		puts("Wrong number of arguments... aborting...");
		return 1;
	}


	/* run urdutag_file */
	urdutag_file(argv[1], argv[2], argv[3]);


	return 0;
}