#include <stdio.h>
#include <stdlib.h>
#include "andrew.h"
#include "commandline.h"
#include "unicode.h"
#include "tagfile.h"
#include "comparetag.h"


/* this comparison algorithm assumes:
   a) the benchmark is free of ambiguity (looks only at 1st tag from that file)
   b) the tokenisation is identical
   
   returns 0 for all OK, 1 for a read error, 2 for a write error, and exits for
   tokenisation mismatch
   */

int comparetag(const char *benchmark_filename, const char *checkme_filename,
			   const char *error_filename, FILE *report)
{
	char thiswordcorrect;
	int i;
	unichar uniNULL[] =  { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	unichar uni0d0a[] =  { 0x000d, 0x000a, 0x0000 };

	float tokencount = 0, errorcount = 0, tagcount = 0;

	FILE *benchmark;
	FILE *checkme;
	FILE *error;

	token *bmword;
	token *chword;


	/* open the source files and check for (then discard) directionality character */
	if (!(benchmark = fopen(benchmark_filename, "rb")))
	{
		cl_error_file_open(benchmark_filename);
		fcloseall();
		return 1;
	}
	if (!( ucheckdir(benchmark) ))
	{
		fputs("Benchmark file not recognised as Unicode, comparetag aborts.", report);
		fcloseall();
		return 1;
	}
	if (!(checkme = fopen(checkme_filename, "rb")))
	{
		cl_error_file_open(checkme_filename);
		fcloseall();
		return 1;
	}
	if (!( ucheckdir(checkme) ))
	{
		fputs("Scrutinised file not recognised as Unicode, comparetag aborts.", report);
		fcloseall();
		return 1;
	}

	/* open the error storage file and insert RIGHTWAY */
	if ( !(error = fopen(error_filename, "wb")) )
	{
		cl_error_file_open(error_filename);
		fcloseall();
		return 2;
	}
	if (fputuc(RIGHTWAY, error) == UERR)
	{
		cl_error_file_write(error_filename);
		fcloseall();
		return 2;
	}

	while (1)
	{
		/* load a line from benchmark; if it fails, break (end-of-file). */
		if ( ! (bmword = load_token(benchmark)) )
			break;


		/* load a line from checkme; if it fails, abort everything */
		if ( ! (chword = load_token(checkme)) )
		{
			printf("Tokenisation match broke down after %.f tokens!\n", tokencount);
			puts("Program aborts.");
			fcloseall();
			exit(1);
		}

		/* check that the tokens are identical. If they are not, abort everything. */
		if ( ! ustrident(bmword->wordform, chword->wordform) )
		{
			printf("Tokenisation match broke down after %.f tokens!\n", tokencount);
			puts("Program aborts.");
			fcloseall();
			exit(1);
		}

		/* increase the token count */
		tokencount++;
		/* but only if it's not SGML */
		if (ustrident(bmword->tag[0], uniNULL) && ustrident(chword->tag[0], uniNULL))
			tokencount--;


		/* examine each tag, ustridenting with the correct tag. */
		thiswordcorrect = FALSE;

		for ( i = 0 ; i < TAGSMAX  && (chword->tag[i][0]) ; i++ )
			if (ustrident(bmword->tag[0], chword->tag[i]))
				thiswordcorrect = TRUE;


		/* how many tags are there? After last for loop we can add i to tagcount */
		tagcount += i;

		/* if no correct tag is found... */
		if (thiswordcorrect == FALSE)
		{
			/* write that line to error */
			if (write_token(chword, error))
			{
				fcloseall();
				return 2;
			}
			if (write_token(bmword, error))
			{
				fcloseall();
				return 2;
			}
			if (fputus(uni0d0a, error) == EOF)
			{
				fcloseall();
				return 2;
			}


			/* and increment the count of errors */
			errorcount++;
		}

		/* free the words */

		free(bmword);
		free(chword);
	}



	/* output to "report" the summary of what has been found out */
	fprintf(report, "\nComparetag report on file %s:\n", checkme_filename);
	fprintf(report, "\nCorrectness:  %f%%", ((tokencount - errorcount) / tokencount) * 100);
	fprintf(report, "\nAmbiguity:    %f\n",  tagcount / tokencount);


	/* close all 3 files */
	if (fclose(benchmark))
	{
		cl_error_file_close(benchmark_filename);
		fcloseall();
		return 1;
	}
	if (fclose(checkme))
	{
		cl_error_file_close(checkme_filename);
		fcloseall();
		return 1;
	}
	if (fclose(error))
	{
		cl_error_file_close(error_filename);
		fcloseall();
		return 1;
	}
	return 0;
}