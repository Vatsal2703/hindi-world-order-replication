#include <stdio.h>
#include <stdlib.h>
#include "andrew.h"
#include "unicode.h"
#include "tagfile.h"
#include "lexicon.h"
#include "u1_tagset.h"
#include "urdutag.h"

void urdutag_file(const char *input_filename, const char *output_filename,
				  const char *lexicon_filename)
{
	FILE *source;
	FILE *dest;

	entry *lexicon;

	token *word;


	/* open the source file and check for (then discard) directionality character */
	if (!(source = fopen(input_filename, "rb")))
	{
		puts("Error opening original file!");
		fcloseall();
		return;
	}
	if (!( ucheckdir(source) ))
	{
		fputs("Specified source file not recognised as Unicode!", stderr);
		fcloseall();
		return;
	}

	/* open file to write, insert directionality character */
	if( !(dest = fopen(output_filename, "wb")) )
	{
		puts("Error opening processed file!");
		fcloseall();
		return;
	}
	if ( fputuc( RIGHTWAY , dest) == UERR )
	{
		puts("Error writing to processed file!");
		fcloseall();
		return;
	}


	if (! ( lexicon = load_lexicon(lexicon_filename) ) )
		return;

	while (1)
	{
		/* read a line */
		if ( ! (word = load_token(source)) )
			break;


		/* urdutag that token IF it has no tags already */

		if (word->tag[0][0] == 0x0000)
			urdutag(lexicon, word);


		/* if split-signal tag returned , perform the special actions */
		if ( word->tag[0][0] == 0x0053 && word->tag[0][1] == 0x0050 )
			word = do_the_splits(word, lexicon, dest);



		/* write the line to file */
		if (write_token(word, dest))
			break;

		free(word);
	}

	/* check "word". If it is still alloc'd, then the loop broke prematurely */
	/* and "word" will need freeing. */
	if (word)
		free(word);

	/* free the lexicon */
	free_lexicon(lexicon);


	/* close read and write files */
	if (fclose(source) < 0)
	{
		puts("Error closing original file!");
		fcloseall();
		return;
	}
	if (fclose(dest) < 0)
	{
		puts("Error closing processed file!");
		fcloseall();
		return;
	}
}



void urdutag(entry *lexicon, token *word)
{
	int i, j;

	entry *rightentry;

	unichar tempword[WORDLENGTH];

	unichar respA10[USERLENGTH] = { 0x0041, 0x0031, 0x0030, 0x0000 };
	unichar respA50[USERLENGTH] = { 0x0041, 0x0035, 0x0030, 0x0000 };
	unichar respA90[USERLENGTH] = { 0x0041, 0x0039, 0x0030, 0x0000 };

	static char before = FALSE;
	static char *c_pointers_tagset[TAGSETSIZE];
	static unichar tagset[TAGSETSIZE][TAGLENGTH];


	if ( ! before )
	{
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
		before = TRUE;
	}


	/* remove vowels from a temporary copy of the word for lookup */
	ustrcpy(tempword, word->wordform);
	disemvowel(tempword);



	/* if the word has the special al-tag, do lexical lookup */
	/* if no joy, assign the set of words that can follow al & return */
	/* this is for when it is called by do_the_splits, rather than urdutag */
	// do a concordance of AL and see what does follow it... improve this set

	if(ustrident(word->tag[0], tagset[SPAL]))
	{
		clear_tags(word);
		if (rightentry = find_entry(tempword, lexicon) )
			assign_tags(word, rightentry);
		else
		{
			ustrcpy(word->tag[0], tagset[JJU]);
			ustrcpy(word->tag[1], tagset[NNUM1N]);
			ustrcpy(word->tag[2], tagset[NNUM1O]);
			ustrcpy(word->tag[3], tagset[NNUM2N]);
			ustrcpy(word->tag[4], tagset[NNUM2O]);
			ustrcpy(word->tag[5], tagset[NNUF1N]);
			ustrcpy(word->tag[6], tagset[NNUF1O]);
			ustrcpy(word->tag[7], tagset[NNUF2N]);
			ustrcpy(word->tag[8], tagset[NNUF2N]);
		}
		/* do not change the resptag */
		return;
	}



	/* look it up in the lexicon - if found, assign code and return */
	if (rightentry = find_entry(tempword, lexicon) )
	{
		assign_tags(word, rightentry);
			/* What if a special-process tag is returned from lex?     */
			/* Do nothing: split signal is handled by calling function */
		ustrcpy(word->resp, respA10);
		return;
	}



	/* analyse chartypes */
	
	/* if all chars are numerals (Indo-Arabic, Arabic or Euro), JDNU. */
	ustrcpy(word->tag[0], tagset[JDNU]);

	for ( i = 0 ; i < WORDLENGTH && word->wordform[i] ; i++ )
	{
		if ( word->wordform[i] < 0x0030 ||
			(word->wordform[i] > 0x0039 && word->wordform[i] < 0x0660) ||
			(word->wordform[i] > 0x0669 && word->wordform[i] < 0x06f0) ||
			 word->wordform[i] > 0x06f9
			)
			word->tag[0][0] = 0;
	}
	/* if JDNU was not wiped, return with resp-tag A50 */
	if (word->tag[0][0])
	{
		ustrcpy(word->resp, respA50);
		return;
	}


	/* if any characters are non-Arabic, FX */
	for ( i = 0 ; i < WORDLENGTH  && word->wordform[i] ; i++ )
	{
		if ( word->wordform[i] < 0x0600 || word->wordform[i] > 0x06ff )
			ustrcpy(word->tag[0], tagset[FX]);
	}

	/* if a tag has been assigned, return with resp-tag A50 */
	if (word->tag[0][0])
	{
		ustrcpy(word->resp, respA50);
		return;
	}



	/* analyse morphology: separate func which operates directly on the token structure */
	urduanalyse(word, tagset);


	if (word->tag[0][0])
		/* i.e. if some tag or other has been assigned     */
		/* don't add a resp tag - urduanalyse does this */
		return;



	/* if all else fails and no tag has been given, apply the default set of tags */
	ustrcpy(word->tag[0],  tagset[JJU]);
	ustrcpy(word->tag[1],  tagset[NNUM1N]);
	ustrcpy(word->tag[2],  tagset[NNUM1O]);
	ustrcpy(word->tag[3],  tagset[NNUM1V]);
	ustrcpy(word->tag[4],  tagset[NNUM2N]);
	ustrcpy(word->tag[5],  tagset[NNUF1N]);
	ustrcpy(word->tag[6],  tagset[NNUF1O]);
	ustrcpy(word->tag[7],  tagset[NNUF1V]);
	ustrcpy(word->tag[8],  tagset[RR]);
	ustrcpy(word->tag[9],  tagset[VV0]);
	ustrcpy(word->tag[10], tagset[VVIT1]);

	/* assign the correct resptag */
	ustrcpy(word->resp, respA90);
}



void urduanalyse(token *word, unichar tagset[TAGSETSIZE][TAGLENGTH])
{
	int i;

	unichar tempword[WORDLENGTH];

	unichar respA30[USERLENGTH] = { 0x0041, 0x0033, 0x0030, 0x0000 };
	unichar respA70[USERLENGTH] = { 0x0041, 0x0037, 0x0030, 0x0000 };



	ustrcpy(tempword, word->wordform);
	disemvowel(tempword);

	for ( i = 0 ; tempword[i+1] ; i++ )
		/* scroll [i] to point at final char in string (prior to NULL) */
		;


	switch (tempword[i])
	{
	case 0x0627:
		switch (tempword[i-1])
		{
		case 0x062a:
			/* -tA : VVTM1N */
			ustrcpy(word->tag[0], tagset[VVTM1N]);
			break;
		case 0x0646:
			/* -nA : VVNM1N */
			ustrcpy(word->tag[0], tagset[VVNM1N]);
			break;
		case 0x067e:
			/* -pA : NNMM1N  */
			ustrcpy(word->tag[0], tagset[NNMM1N]);
			break;
		case 0x06cc:
			/* -YA : NNMF1N, NNMF1O, NNMF1V, VVYM1N */
			ustrcpy(word->tag[0], tagset[NNMF1N]);
			ustrcpy(word->tag[1], tagset[NNMF1O]);
			ustrcpy(word->tag[2], tagset[NNMF1V]);
			ustrcpy(word->tag[3], tagset[VVYM1N]);
			break;
		default:
			/* -A with no other clues: NNMM1N, JJM1N, VVYM1N */
			ustrcpy(word->tag[0], tagset[NNMM1N]);
			ustrcpy(word->tag[1], tagset[JJM1N]);
			ustrcpy(word->tag[2], tagset[VVYM1N]);
			break;
		}
		break;

	case 0x062a:
		if (tempword[i-1] == 0x06cc)
		{
			/* -Yt: apply NNUF1N, NNUF1O, NNUF1V */
			ustrcpy(word->tag[0], tagset[NNUF1N]);
			ustrcpy(word->tag[1], tagset[NNUF1O]);
			ustrcpy(word->tag[2], tagset[NNUF1V]);
		}
		break;

	case 0x0646:
		if((tempword[i-1] == 0x0627 &&
			tempword[i-2] == 0x062a &&
			tempword[i-3] == 0x0633 )  ||
		   (tempword[i-1] == 0x067e )	)
		{
			/* -stAn or -pn: apply NNUM1N, NNUM1O, NNUM1V, NNUM2N */
			ustrcpy(word->tag[0], tagset[NNUM1N]);
			ustrcpy(word->tag[1], tagset[NNUM1O]);
			ustrcpy(word->tag[2], tagset[NNUM1V]);
			ustrcpy(word->tag[3], tagset[NNUM2N]);
		}
		break;

	case 0x0648:
		if (tempword[i-1] == 0x062a &&
			tempword[i-2] == 0x06cc )
		{
			/* -YtV : NNUF2V */
			ustrcpy(word->tag[0], tagset[NNUF2V]);
			break;
		}
		if((tempword[i-1] == 0x0646 &&
			tempword[i-2] == 0x0627 &&
			tempword[i-3] == 0x062a &&
			tempword[i-4] == 0x0633 )	||
		   (tempword[i-1] == 0x0646 &&
			tempword[i-2] == 0x067e )	)
		{
			/* -stAnV or -pnV: apply NNUM2V */
			ustrcpy(word->tag[0], tagset[NNUM2V]);
			break;
		}
		if((tempword[i-1] == 0x0679 &&
			tempword[i-2] == 0x0648 &&
			tempword[i-3] == 0x0627 )	||
		   (tempword[i-1] == 0x0679 &&
		    tempword[i-2] == 0x06c1 &&
			tempword[i-3] == 0x0627 )	||
		   (tempword[i-1] == 0x06c1 &&
			tempword[i-2] == 0x0627 &&
			tempword[i-3] == 0x06af )	)
		{
			/* -AVTV, -AhTV, or -gAHV : NNUF2V */
			ustrcpy(word->tag[0], tagset[NNUF2V]);
			break;
		}
		/* "else": just -V : therefore assign NNMM2V, NNMF2V, NNUM2V, NNUF2V, VVST2, VVIT2 */
		ustrcpy(word->tag[0], tagset[NNMM2V]);
		ustrcpy(word->tag[1], tagset[NNMF2V]);
		ustrcpy(word->tag[2], tagset[NNUM2V]);
		ustrcpy(word->tag[3], tagset[NNUF2V]);
		ustrcpy(word->tag[4], tagset[VVST2]);
		ustrcpy(word->tag[5], tagset[VVIT2]);
		break;

	case 0x064b:
		/* word ends in tanvYn, therefore assign RR */
		ustrcpy(word->tag[0], tagset[RR]);
		break;

	case 0x0679:
		if((tempword[i-1] == 0x0648 &&
			tempword[i-2] == 0x0627 )	||
		   (tempword[i-1] == 0x06c1 &&
			tempword[i-2] == 0x0627 )	)
		{
			/* -AVT or -AhT : NNUF1N, NNUF1O, NNUF1V */
			ustrcpy(word->tag[0], tagset[NNUF1N]);
			ustrcpy(word->tag[1], tagset[NNUF1O]);
			ustrcpy(word->tag[2], tagset[NNUF1V]);
		}
		break;


	/**************************************************************************************/
	/*                     BEGIN THE ONES WITH ~ AS THEIR LAST CHAR                       */

	case 0x06ba:
		switch (tempword[i-1])
		{
		case 0x0627: /* -A~ */
			if (tempword[i-2] == 0x0648)
			{
				/* -VA~ : assign JDNM1N */
				ustrcpy(word->tag[0], tagset[JDNM1N]);
			}
			else if (tempword[i-2] == 0x06cc)
			{
				/* -YA~ : assign NNMF2N */
				ustrcpy(word->tag[0], tagset[NNMF2N]);
			}
			break;

		case 0x0648: /* -V~ */
			if((tempword[i-2] == 0x0646 &&
				tempword[i-3] == 0x0627 &&
				tempword[i-4] == 0x062a &&
				tempword[i-5] == 0x0633	)	||
			   (tempword[i-2] == 0x0646 &&
				tempword[i-3] == 0x067e	)	)
			{
				/* -stAnV~ or -pnV~ : assign NNUM2O */
				ustrcpy(word->tag[0], tagset[NNUM2O]);
				break;
			}
			if((tempword[i-2] == 0x062a &&
				tempword[i-3] == 0x06cc )	||
			   (tempword[i-2] == 0x0679 &&
				tempword[i-3] == 0x0648 &&
				tempword[i-4] == 0x0627 )	||
			   (tempword[i-2] == 0x0679 &&
				tempword[i-3] == 0x06c1 &&
				tempword[i-4] == 0x0627 )	||
			   (tempword[i-2] == 0x06c1 &&
				tempword[i-3] == 0x0627 &&
				tempword[i-4] == 0x06af )	)
			{
				/* -YtV~, -AVTV~, -AhTV~ or -gAhV~ : assign NNUF2O */
				ustrcpy(word->tag[0], tagset[NNUF2O]);
				break;
			}
			if (tempword[i-2] == 0x06cc)
			{
				/* -YV~ : assign NNMF2O */
				ustrcpy(word->tag[0], tagset[NNMF2O]);
				break;
			}
			/* "else" it's just -V~ : assign NNMM2O, NNUM2O, NNUF2O, VVSM1 */
			ustrcpy(word->tag[0], tagset[NNMM2O]);
			ustrcpy(word->tag[1], tagset[NNUM2O]);
			ustrcpy(word->tag[2], tagset[NNUF2O]);
			ustrcpy(word->tag[3], tagset[VVSM1]);
			break;

		case 0x06cc: /* -Y~ */
			if (tempword[i-2] == 0x062a &&
				tempword[i-3] == 0x06cc 	)
			{
				/* -YtY~ : assign NNUF2N */
				ustrcpy(word->tag[0], tagset[NNUF2N]);
				break;
			}
			if (tempword[i-2] == 0x062a)
			{
				/* -tY~ : assign VVTF2N */
				ustrcpy(word->tag[0], tagset[VVTF2N]);
				break;
			}
			if (tempword[i-2] == 0x0648)
			{
				/* -VY~ : assign JDNM1O JDNM2N JDNM2O JDNF1N JDNF1O JDNF2N JDNF2O */
				ustrcpy(word->tag[0], tagset[JDNM1O]);
				ustrcpy(word->tag[1], tagset[JDNM2N]);
				ustrcpy(word->tag[2], tagset[JDNM2O]);
				ustrcpy(word->tag[3], tagset[JDNF1N]);
				ustrcpy(word->tag[4], tagset[JDNF1O]);
				ustrcpy(word->tag[5], tagset[JDNF2N]);
				ustrcpy(word->tag[6], tagset[JDNF2O]);
				break;
			}
			if((tempword[i-2] == 0x0679 &&
				tempword[i-3] == 0x0648 &&
				tempword[i-4] == 0x0627 )	||
			   (tempword[i-2] == 0x0679 &&
				tempword[i-3] == 0x06c1 &&
				tempword[i-4] == 0x0627 )	||
			   (tempword[i-2] == 0x06c1 &&
				tempword[i-3] == 0x0627 &&
				tempword[i-4] == 0x06af )	)
			{
				/* -AVTY~, -AhTY~ or -gAhY~: assign NNUF2N */
				ustrcpy(word->tag[0], tagset[NNUF2N]);
				break;
			}
			/* it's just -Y~, so assign: NNUF2N, VVSM2, VVSV2, VVYF2N */
			ustrcpy(word->tag[0], tagset[NNUF2N]);
			ustrcpy(word->tag[1], tagset[VVSM2]);
			ustrcpy(word->tag[2], tagset[VVSV2]);
			ustrcpy(word->tag[3], tagset[VVYF2N]);
			break;

		default:
			/* no pattern for words just randomly ending in nVni GVnA */
			break;
		}
		break;

	/*                     END THE ONES WITH ~ AS THEIR LAST CHAR                         */
	/**************************************************************************************/


	case 0x06c1:
		if (tempword[i-1] == 0x0627 &&
			tempword[i-2] == 0x06af )
		{
			/* -gAh : assign NNUF1N, NNUF1O, NNUF1V */
			ustrcpy(word->tag[0], tagset[NNUF1N]);
			ustrcpy(word->tag[1], tagset[NNUF1O]);
			ustrcpy(word->tag[2], tagset[NNUF1V]);
		}
		else
			/* theoretically check for -Yh, but this is masc marked, just like -h. */
			/* therefore, have found -h or -Yh : assign NNMM1N */
			ustrcpy(word->tag[0], tagset[NNMM1N]);
		break;

	case 0x06cc:
		if (tempword[i-1] == 0x062a)
		{
			/* -tY : assign VVTF1N, VVTF1O, VVTF2N, VVTF2O */
			ustrcpy(word->tag[0], tagset[VVTF1N]);
			ustrcpy(word->tag[1], tagset[VVTF1O]);
			ustrcpy(word->tag[2], tagset[VVTF2N]);
			ustrcpy(word->tag[3], tagset[VVTF2O]);
			break;
		}
		if (tempword[i-1] == 0x0646)
		{
			/* -nY : assign VVNF1, VVNF2 */
			ustrcpy(word->tag[0], tagset[VVNF1]);
			ustrcpy(word->tag[1], tagset[VVNF2]);
			break;
		}
		/* "else": just -Y : therefore assign NNMF1N, NNMF1O, NNMF1V, JJF1N, JJF1O, JJF2N, */
		/*                                    JJF2O, VVYF1N, VVYF1O, VVYF2N, VVYF2O        */
		ustrcpy(word->tag[0], tagset[NNMF1N]);
		ustrcpy(word->tag[1], tagset[NNMF1O]);
		ustrcpy(word->tag[2], tagset[NNMF1V]);
		ustrcpy(word->tag[3], tagset[JJF1N]);
		ustrcpy(word->tag[4], tagset[JJF1O]);
		ustrcpy(word->tag[5], tagset[JJF2N]);
		ustrcpy(word->tag[6], tagset[JJF2O]);
		ustrcpy(word->tag[7], tagset[VVYF1N]);
		ustrcpy(word->tag[8], tagset[VVYF1O]);
		ustrcpy(word->tag[9], tagset[VVYF2N]); /* F2N without ~ ??????? */
		ustrcpy(word->tag[10], tagset[VVYF2O]);
		break;

	case 0x06d2:
		switch (tempword[i-1])
		{
		case 0x0626:
			if (tempword[i-2] == 0x0627)
			{
				/* -A'E : assign NNUF1N, NNUF1O, NNUF1V */
				ustrcpy(word->tag[0], tagset[NNUF1N]);
				ustrcpy(word->tag[1], tagset[NNUF1O]);
				ustrcpy(word->tag[2], tagset[NNUF1V]);
			}
			else
			{
				/* -'E : assign VVIA, NNMM1O, NNMM1V, NNMM2N */
				ustrcpy(word->tag[0], tagset[VVIA]);
				ustrcpy(word->tag[1], tagset[NNMM1O]);
				ustrcpy(word->tag[2], tagset[NNMM1V]);
				ustrcpy(word->tag[3], tagset[NNMM2N]);
			}
			break;
		case 0x062a:
			/* -tE : assign VVTM1O, VVTM2N, VVTM2O */
			ustrcpy(word->tag[0], tagset[VVTM1O]);
			ustrcpy(word->tag[1], tagset[VVTM2N]);
			ustrcpy(word->tag[2], tagset[VVTM2O]);
			break;
		case 0x0646:
			/* -nE : assign VVNM1O, VVNM2 */
			ustrcpy(word->tag[0], tagset[VVNM1O]);
			ustrcpy(word->tag[1], tagset[VVNM2]);
			break;
		case 0x06cc:
			if (tempword[i-2] == 0x0626)
			{
				/* -'YE : assign VVIA */
				ustrcpy(word->tag[0], tagset[VVIA]);
				break;
			}
			else
				;
			/* drop-thru here purposeful so the ELSE of the previous IF is "GOTO DEFAULT" */
		default:
			/* it's just -E : assign NNMM1O, NNMM1V, NNMM2N, JJM1O, JJM2N, JJM2O, */
			/*                       VVYM1O, VVYM2N, VVYM2O, VVST1, VVSV1, RRJ    */
			ustrcpy(word->tag[0], tagset[NNMM1O]);
			ustrcpy(word->tag[1], tagset[NNMM1V]);
			ustrcpy(word->tag[2], tagset[NNMM2N]);
			ustrcpy(word->tag[3], tagset[JJM1O]);
			ustrcpy(word->tag[4], tagset[JJM2N]);
			ustrcpy(word->tag[5], tagset[JJM2O]);
			ustrcpy(word->tag[6], tagset[VVYM1O]);
			ustrcpy(word->tag[7], tagset[VVYM2N]);
			ustrcpy(word->tag[8], tagset[VVYM2O]);
			ustrcpy(word->tag[9], tagset[VVST1]);
			ustrcpy(word->tag[10], tagset[VVSV1]);
			ustrcpy(word->tag[11], tagset[RRJ]);
			break;
		}
		break;

	default:
		break;
	}

	/* if a tag has been given, assign resptag A30 and return */
	/* THIS MEANS ALL "AL-" WORDS THAT AREN'T AL/NNU... MUST BE IN LEXICON !!! */

	if (word->tag[0][0])
	{
		ustrcpy(word->resp, respA30);
		return;
	}



	/* check for "al" */
	if ( word->wordform[0] == 0x0627 && word->wordform[1] == 0x0644 )
	{
		/* assign SPAL, add a resptag (A70) and return */
		ustrcpy(word->tag[0], tagset[SPAL]);
		ustrcpy(word->resp, respA70);
		return;
	}

	/* if even looking for "al" has failed, give up */
}



token *do_the_splits(token *word, entry *lexicon, FILE *dest)
{
	unichar u_SPAL[]    = {0x53, 0x50, 0x41, 0x4c, 0x00};
	unichar u_SPGUNA[]  = {0x53, 0x50, 0x47, 0x55, 0x4e, 0x41, 0x00};
	unichar u_SPKE[]    = {0x53, 0x50, 0x4b, 0x45, 0x00};
	unichar u_SPXYM[]   = {0x53, 0x50, 0x58, 0x59, 0x4d, 0x00};
	unichar u_SPXDYM[]  = {0x53, 0x50, 0x58, 0x44, 0x59, 0x4d, 0x00};
	unichar u_SPXHYM[]  = {0x53, 0x50, 0x58, 0x48, 0x59, 0x4d, 0x00};
	unichar u_SPHY[]    = {0x53, 0x50, 0x48, 0x59, 0x00};
	unichar u_SPHDY[]   = {0x53, 0x50, 0x48, 0x44, 0x59, 0x00};
	unichar u_SPLIE[]	= {0x53, 0x50, 0x4c, 0x49, 0x45, 0x00};

	unichar t_AL[]      = {0x41, 0x4c, 0x00};
	unichar t_JDNUC[]   = {0x4a, 0x44, 0x4e, 0x55, 0x43, 0x00};
	unichar t_IIC[]		= {0x49, 0x49, 0x43, 0x00};
	unichar t_XHC[]		= {0x58, 0x48, 0x43, 0x00};
	unichar t_IIM1N[]	= {0x49, 0x49, 0x4d, 0x31, 0x4e, 0x00};
	unichar t_RR[]		= {0x52, 0x52, 0x00};

	token *word1;
	token *word2;


	/* copy the token*/
	word1 = word;
	word2 = copy_token(word);


	if (ustrident(word->tag[0], u_SPAL))
	{
		/* 1st copy: wipe all but first two characters */
		word1->wordform[2] = 0;
		/* give it the tag AL */
		clear_tags(word1);
		ustrcpy(word1->tag[0], t_AL);
		/* write it to file */
		if (write_token(word1, dest))
		{
			puts("Error! Error! writing split form to file... aborting now...");
			fcloseall();
			exit(1);
		}

		/* 2nd copy: trim off first two characters */
		ustrimstart(word2->wordform, 2);
		/* 2nd copy: resubmit that word, still with SPAL */
		urdutag(lexicon, word2);
	}
	else if (ustrident(word->tag[0], u_SPGUNA))
	{
		/* 1st copy: shorten by three chars (length of gunaa) */
		ustrimend(word1->wordform, 3);
		/* give it the JDNUC tag */
		clear_tags(word1);
		ustrcpy(word1->tag[0], t_JDNUC);
		/* write it to file */
		if (write_token(word1, dest))
		{
			puts("It's all gone horribly wrong... aborting program!");
			fcloseall();
			exit(1);
		}

		/* 2nd copy: remove all but the three last characters */
		ustrimstart(word2->wordform, (ustrlen(word2->wordform) - 3));
		/* 2nd copy: resubmit that word, having nulled its tags */
		clear_tags(word2);
		urdutag(lexicon, word2);
	}
	else if (ustrident(word->tag[0], u_SPKE))
	{
		/* 1st copy: shorten by 1 char */
		ustrimend(word1->wordform, 1);
		/* clear tags, resubmit to urdutag */
		clear_tags(word1);
		urdutag(lexicon, word1);
		/* write it to file */
		if (write_token(word1, dest))
		{
			puts("I didn't want to be an analysis program, you know.");
			puts("I wanted to be... a lumberjack!");
			fcloseall();
			exit(1);
		}

		/* 2nd copy: remove all but the last 1 character */
		ustrimstart(word2->wordform, (ustrlen(word2->wordform) - 1));		
		/* 2nd copy: give it the IIC tag, clearing tags first */
		clear_tags(word2);
		ustrcpy(word2->tag[0], t_IIC);
	}
	else if (ustrident(word->tag[0], u_SPXHYM ))
	{
		/* 1st copy: shorten by three chars */
		ustrimend(word1->wordform, 3);
		/* clear tags, resubmit to urdutag */
		clear_tags(word1);
		urdutag(lexicon, word1);
		/* write it to file */
		if (write_token(word1, dest))
		{
			puts("I'm going to end it all... Goodbye cruel world!");
			fcloseall();
			exit(1);
		}

		/* 2nd copy: remove all but the three last characters */
		ustrimstart(word2->wordform, (ustrlen(word2->wordform) - 3));
		/* 2nd copy: give tags XHC and IIC */
		clear_tags(word2);
		ustrcpy(word2->tag[0], t_IIC);
		ustrcpy(word2->tag[1], t_XHC);
	}
	else if (ustrident(word->tag[0], u_SPHY))
	{
		/* 1st copy: shorten by 1 char */
		ustrimend(word1->wordform, 1);
		/* clear tags, resubmit to urdutag */
		clear_tags(word1);
		urdutag(lexicon, word1);
		/* write it to file */
		if (write_token(word1, dest))
		{
			puts("It's not working, OKAY? Sod off and leave me alone.");
			fcloseall();
			exit(1);
		}

		/* 2nd copy: remove all but the last 1 character */
		ustrimstart(word2->wordform, (ustrlen(word2->wordform) - 1));
		/* 2nd copy: give tag XHC */
		clear_tags(word2);
		ustrcpy(word2->tag[0], t_XHC);
	}
	else if (ustrident(word->tag[0], u_SPHDY))
	{
		/* 1st copy: lose all but first two characters */
		ustrimend(word1->wordform, 2);
		/* clear tags, resubmit to urdutag */
		clear_tags(word1);
		urdutag(lexicon, word1);
		/* write it to file */
		if (write_token(word1, dest))
		{
			puts("OK, the good news is your program was working really, really well.");
			puts("The bad news is it ain't no more.");
			fcloseall();
			exit(1);
		}

		/* 2nd copy: remove all but final 2 chars */
		ustrimstart(word2->wordform, (ustrlen(word2->wordform) - 2));
		/* 2nd copy: give tag XHC */
		clear_tags(word2);
		ustrcpy(word2->tag[0], t_XHC);
	}
	else if (ustrident(word->tag[0], u_SPLIE))
	{
		/* 1st copy: wipe all but first two characters */
		word1->wordform[2] = 0;
		/* make end char to E to make sure it's kE not kY  - it may have been disemvowelled! */
		word1->wordform[1] = 0x06d2;
		/* give it the tag IIM1N */
		clear_tags(word1);
		ustrcpy(word1->tag[0], t_IIM1N);
		/* write it to file */
		if (write_token(word1, dest))
		{
			puts("I'm a poor ickle anawysis pwogwam an I don't feel vewy well....");
			fcloseall();
			exit(1);
		}

		/* 2nd copy: remove first 2 chars */
		ustrimstart(word2->wordform, 2);
		/* 2nd copy: give tag RR */
		clear_tags(word2);
		ustrcpy(word2->tag[0], t_RR);
	}
	/* if none of these apply, just leave the whole thing the same and return as-is */

	free(word1);

	/* return word 2 to be written by calling function */
	return word2;
}



void disemvowel(unichar *string)
{
	int i, j;

	unichar itnaa[] = { 0x0627, 0x0650, 0x062a, 0x0646, 0x0627, 0x0000 };
	unichar utnaa[] = { 0x0627, 0x064f, 0x062a, 0x0646, 0x0627, 0x0000 };
	unichar itnee[] = { 0x0627, 0x0650, 0x062a, 0x0646, 0x06d2, 0x0000 };
	unichar utnee[] = { 0x0627, 0x064f, 0x062a, 0x0646, 0x06d2, 0x0000 };
	unichar itnii[] = { 0x0627, 0x0650, 0x062a, 0x0646, 0x06cc, 0x0000 };
	unichar utnii[] = { 0x0627, 0x064f, 0x062a, 0x0646, 0x06cc, 0x0000 };
	unichar is[]    = { 0x0627, 0x0650, 0x0633, 0x0000 };
	unichar us[]    = { 0x0627, 0x064f, 0x0633, 0x0000 };
	unichar in[]    = { 0x0627, 0x0650, 0x0646, 0x0000 };
	unichar un[]    = { 0x0627, 0x064f, 0x0646, 0x0000 };
	unichar idhar[] = { 0x0627, 0x0650, 0x062f, 0x06be, 0x0631, 0x0000 };
	unichar udhar[] = { 0x0627, 0x0650, 0x062f, 0x06be, 0x0631, 0x0000 };


	/* this bit checks for the words that should NOT be de-voweled */
	if (   ustrident(string, itnaa)
		|| ustrident(string, utnaa)
		|| ustrident(string, itnee)
		|| ustrident(string, utnee)
		|| ustrident(string, itnii)
		|| ustrident(string, utnii)
		|| ustrident(string, is)
		|| ustrident(string, us)
		|| ustrident(string, in)
		|| ustrident(string, un)
		|| ustrident(string, idhar)
		|| ustrident(string, udhar)
		)
		return;

	/* change initial docashmii he to choTii he */
	/* this needs to be done to every word except hii and hii~ as XHC and IIC */
	/* but the clitics are not disemvowelled ANYWAY (see "do the splits") */
	/* so... */

	if (*string == 0x06be)
		*string = 0x06c1;

	for ( i = 0 ; *(string+i) ; i++ )
	{
		/* remove diacritics */
		if ( *(string+i) > 0x064d && *(string+i) < 0x0652)
		{
			for ( j = i ; *(string+j) ; j++ )
			{
				*(string+j) = *(string+j+1);
				i--;
				continue;
			}
		}

		/* change non-final baRii ye to choTii ye */
		if ( *(string+i) == 0x06d2 && *(string+i+1) != 0x0000 )
			*(string+i) = 0x06cc;
		if ( *(string+i) == 0x06d3 && *(string+i+1) != 0x0000 )
			*(string+i) = 0x0626;

		/* change alif-maksura to choTii ye */
		if ( *(string+i) == 0x0649 )
			*(string+i) = 0x06cc;
		/* change Arabic-ye to choTii ye */
		if ( *(string+i) == 0x064a )
			*(string+i) = 0x06cc;
	}
}