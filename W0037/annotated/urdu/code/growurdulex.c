#include <stdio.h>
#include <string.h>
#include "andrew.h"
#include "commandline.h"
#include "unicode.h"
#include "tagfile.h"
#include "lexicon.h"
#include "u1_tagset.h"
#include "urdutag.h"


main(int argc, char *argv[])
{
	entry *lex;
	entry *current;

	char *input_filename;
	char output_filename[250];

	int i, j;

	char *c_pointers_tagset[TAGSETSIZE];
	unichar tagset[TAGSETSIZE][TAGLENGTH];


	/* create the tagset strings in an array */

	initialise_tagset_pointer_array(c_pointers_tagset);
	for ( i = 0 ; i < TAGSETSIZE ; i++ )
	{
		for ( j = 0 ; j < TAGLENGTH ; j++ )
		{
			tagset[i][j] = *(c_pointers_tagset[i]+j);
			if ( ! tagset[i][j] )
				break;
		}
	}



	if (argc != 2)
	{
		cl_error_arguments(2, argc);
		return 1;
	}

	input_filename = argv[1];

	strcpy(output_filename, "enr_");
	strcat(output_filename, input_filename);


	if ( ! (lex = load_lexicon(input_filename)) )
	{
		cl_error_file_open(input_filename);
		return 1;
	}


	for ( current = lex ; current ; current = current->next_entry )
	{
		/* disemvowel the word - because it will be disemvowelled when looking for it */

		disemvowel(current->wordform);

		/* marked adjectives */
		if ( entry_tagsinc(current, tagset[JJM1O]) )
		{
			entry_addtag(current, tagset[JJM2N]);
			entry_addtag(current, tagset[JJM2O]);
		}
		if ( entry_tagsinc(current, tagset[JJM2N]) )
		{
			entry_addtag(current, tagset[JJM1O]);
			entry_addtag(current, tagset[JJM2O]);
		}
		if ( entry_tagsinc(current, tagset[JJM2O]) )
		{
			entry_addtag(current, tagset[JJM2N]);
			entry_addtag(current, tagset[JJM1O]);
		}
		if ( entry_tagsinc(current, tagset[JJF1N]) )
		{
			entry_addtag(current, tagset[JJF1O]);
			entry_addtag(current, tagset[JJF2N]);
			entry_addtag(current, tagset[JJF2O]);
		}
		if ( entry_tagsinc(current, tagset[JJF1O]) )
		{
			entry_addtag(current, tagset[JJF1N]);
			entry_addtag(current, tagset[JJF2N]);
			entry_addtag(current, tagset[JJF2O]);
		}
		if ( entry_tagsinc(current, tagset[JJF2N]) )
		{
			entry_addtag(current, tagset[JJF1N]);
			entry_addtag(current, tagset[JJF1O]);
			entry_addtag(current, tagset[JJF2O]);
		}
		if ( entry_tagsinc(current, tagset[JJF2N]) )
		{
			entry_addtag(current, tagset[JJF1N]);
			entry_addtag(current, tagset[JJF1O]);
			entry_addtag(current, tagset[JJF2O]);
		}

		/* marked nouns */
		if ( entry_tagsinc(current, tagset[NNMM1V]) )
		{
			entry_addtag(current, tagset[NNMM1O]);
		}
		if ( entry_tagsinc(current, tagset[NNMM2N]) )
		{
			entry_addtag(current, tagset[NNMM1O]);
			entry_addtag(current, tagset[NNMM1V]);
		}
		if ( entry_tagsinc(current, tagset[NNMF1N]) )
		{
			entry_addtag(current, tagset[NNMF1O]);
		}
		if ( entry_tagsinc(current, tagset[NNMF1O]) )
		{
			entry_addtag(current, tagset[NNMF1N]);
		}
		if ( entry_tagsinc(current, tagset[NNMF1V]) )
		{
			entry_addtag(current, tagset[NNMF1N]);
			entry_addtag(current, tagset[NNMF1O]);
		}

		/* unmarked nouns */
		if ( entry_tagsinc(current, tagset[NNUM1N]) )
		{
			entry_addtag(current, tagset[NNUM1O]);
		}
		if ( entry_tagsinc(current, tagset[NNUM1O]) )
		{
			entry_addtag(current, tagset[NNUM1N]);
		}
		if ( entry_tagsinc(current, tagset[NNUM1V]) )
		{
			entry_addtag(current, tagset[NNUM1N]);
			entry_addtag(current, tagset[NNUM1O]);
		}
		if ( entry_tagsinc(current, tagset[NNUF1N]) )
		{
			entry_addtag(current, tagset[NNUF1O]);
		}
		if ( entry_tagsinc(current, tagset[NNUF1O]) )
		{
			entry_addtag(current, tagset[NNUF1N]);
		}
		if ( entry_tagsinc(current, tagset[NNUF1V]) )
		{
			entry_addtag(current, tagset[NNUF1N]);
			entry_addtag(current, tagset[NNUF1O]);
		}

		/* marked-adj derived adverb */
		if  (   entry_tagsinc(current, tagset[RRJ])
			&& !entry_tagsinc(current, tagset[JDNM1O]) )
		{
			entry_addtag(current, tagset[JJM1O]);
			entry_addtag(current, tagset[JJM2N]);
			entry_addtag(current, tagset[JJM2O]);
		}

		/* infinitive */
		if ( entry_tagsinc(current, tagset[VVNM1O]) )
		{
			entry_addtag(current, tagset[VVNM2]);
		}
		if ( entry_tagsinc(current, tagset[VVNM2]) )
		{
			entry_addtag(current, tagset[VVNM1O]);
		}
		if ( entry_tagsinc(current, tagset[VVNF1]) )
		{
			entry_addtag(current, tagset[VVNF2]);
		}
		if ( entry_tagsinc(current, tagset[VVNF2]) )
		{
			entry_addtag(current, tagset[VVNF1]);
		}

		/* imperfective participle */
		if ( entry_tagsinc(current, tagset[VVTM1O]) )
		{
			entry_addtag(current, tagset[VVTM2N]);
			entry_addtag(current, tagset[VVTM2O]);
		}
		if ( entry_tagsinc(current, tagset[VVTM2N]) )
		{
			entry_addtag(current, tagset[VVTM1O]);
			entry_addtag(current, tagset[VVTM2O]);
		}
		if ( entry_tagsinc(current, tagset[VVTM2O]) )
		{
			entry_addtag(current, tagset[VVTM1O]);
			entry_addtag(current, tagset[VVTM2N]);
		}
		if ( entry_tagsinc(current, tagset[VVTF1N]) )
		{
			entry_addtag(current, tagset[VVTF1O]);
			entry_addtag(current, tagset[VVTF2N]);
			entry_addtag(current, tagset[VVTF2O]);
		}
		if ( entry_tagsinc(current, tagset[VVTF1O]) )
		{
			entry_addtag(current, tagset[VVTF1N]);
			entry_addtag(current, tagset[VVTF2N]);
			entry_addtag(current, tagset[VVTF2O]);
		}
		if ( entry_tagsinc(current, tagset[VVTF2N]) )
		{
			entry_addtag(current, tagset[VVTF1N]);
			entry_addtag(current, tagset[VVTF1O]);
			entry_addtag(current, tagset[VVTF2O]);
		}
		if ( entry_tagsinc(current, tagset[VVTF2O]) )
		{
			entry_addtag(current, tagset[VVTF1N]);
			entry_addtag(current, tagset[VVTF1O]);
			entry_addtag(current, tagset[VVTF2N]);
		}

		/* perfective participle */
		if ( entry_tagsinc(current, tagset[VVYM1O]) )
		{
			entry_addtag(current, tagset[VVYM2N]);
			entry_addtag(current, tagset[VVYM2O]);
		}
		if ( entry_tagsinc(current, tagset[VVYM2N]) )
		{
			entry_addtag(current, tagset[VVYM1O]);
			entry_addtag(current, tagset[VVYM2O]);
		}
		if ( entry_tagsinc(current, tagset[VVYM2O]) )
		{
			entry_addtag(current, tagset[VVYM1O]);
			entry_addtag(current, tagset[VVYM2N]);
		}
		if ( entry_tagsinc(current, tagset[VVYF1N]) )
		{
			entry_addtag(current, tagset[VVYF1O]);
			entry_addtag(current, tagset[VVYF2N]);
			entry_addtag(current, tagset[VVYF2O]);
		}
		if ( entry_tagsinc(current, tagset[VVYF1O]) )
		{
			entry_addtag(current, tagset[VVYF1N]);
			entry_addtag(current, tagset[VVYF2N]);
			entry_addtag(current, tagset[VVYF2O]);
		}
		if ( entry_tagsinc(current, tagset[VVYF2O]) )
		{
			entry_addtag(current, tagset[VVYF1N]);
			entry_addtag(current, tagset[VVYF1O]);
			entry_addtag(current, tagset[VVYF2N]);
		}

	}

	lex = tidy_lexicon(lex);

	save_lexicon(lex, output_filename, 0, NO);

	free_lexicon(lex);

	printf("Lexicon saved, memory cleared... done with %s!\n", output_filename);

	return 0;
}


/* This is how the relevent sections of growurdulex were before my change (the nouns section)

The change was made to stop vocative getting added to nouns for which there was no evidence
they could be vocative.

		if ( entry_tagsinc(current, tagset[NNMM1O]) )
		{
			entry_addtag(current, tagset[NNMM1V]);
		}
		if ( entry_tagsinc(current, tagset[NNMM1V]) )
		{
			entry_addtag(current, tagset[NNMM1O]);
		}
		if ( entry_tagsinc(current, tagset[NNMM2N]) )
		{
			entry_addtag(current, tagset[NNMM1O]);
			entry_addtag(current, tagset[NNMM1V]);
		}
		if ( entry_tagsinc(current, tagset[NNMF1N]) )
		{
			entry_addtag(current, tagset[NNMF1O]);
			entry_addtag(current, tagset[NNMF1V]);
		}
		if ( entry_tagsinc(current, tagset[NNMF1O]) )
		{
			entry_addtag(current, tagset[NNMF1N]);
			entry_addtag(current, tagset[NNMF1V]);
		}
		if ( entry_tagsinc(current, tagset[NNMF1V]) )
		{
			entry_addtag(current, tagset[NNMF1N]);
			entry_addtag(current, tagset[NNMF1O]);
		}

		/* unmarked nouns * /
		if ( entry_tagsinc(current, tagset[NNUM1N]) )
		{
			entry_addtag(current, tagset[NNUM1O]);
			entry_addtag(current, tagset[NNUM1V]);
		}
		if ( entry_tagsinc(current, tagset[NNUM1O]) )
		{
			entry_addtag(current, tagset[NNUM1N]);
			entry_addtag(current, tagset[NNUM1V]);
		}
		if ( entry_tagsinc(current, tagset[NNUM1V]) )
		{
			entry_addtag(current, tagset[NNUM1N]);
			entry_addtag(current, tagset[NNUM1O]);
		}
		if ( entry_tagsinc(current, tagset[NNUF1N]) )
		{
			entry_addtag(current, tagset[NNUF1O]);
			entry_addtag(current, tagset[NNUF1V]);
		}
		if ( entry_tagsinc(current, tagset[NNUF1O]) )
		{
			entry_addtag(current, tagset[NNUF1N]);
			entry_addtag(current, tagset[NNUF1V]);
		}
		if ( entry_tagsinc(current, tagset[NNUF1V]) )
		{
			entry_addtag(current, tagset[NNUF1N]);
			entry_addtag(current, tagset[NNUF1O]);
		}

end removed section */