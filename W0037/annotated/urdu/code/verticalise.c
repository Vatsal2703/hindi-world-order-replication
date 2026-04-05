#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "andrew.h"
#include "unicode.h"
#include "verticalise.h"

/* return values: 		(see verticalise.h)
	SUCCESS 0
	ERROR_OPEN_INPUT
	ERROR_READ_INPUT
	ERROR_CLOSE_INPUT
	ERROR_OPEN_OUTPUT
	ERROR_WRITE_OUTPUT
	ERROR_CLOSE_OUTPUT 
	ERROR_MALLOC
	NONUNICODE (i.e. does not begin with 0xfeff)

This function does not communicate with the user, only with the calling function
All arguments are ASCII. The third must be exactly three characters in length.
Argument checking should be performed by the calling function.

*/
int verticalise(const char *input_filename, const char *output_filename, const char *ascii_user)
{
	/* -------------------------- */
	/* begin variable declaration */ 
	/* -------------------------- */

	unichar current, previous, peekahead;

	char buffer[80];
	unichar linebreak[3] = { 0x00d, 0x00a, 0x0000 } ;
	unichar *unibuffer = NULL;
	unichar *usertag;
	unichar *progtag;
	unichar *resptag = NULL;

	FILE *source;
	FILE *dest;

	/* counters */
	unsigned long sentences = 1, words = 1, wordlength;

	/* program flow control variables */
	int filedone = NO, final = SUCCESS/*, i, spaces*/;

	/* ------------------------ */
	/* end variable declaration */ 
	/* ------------------------ */

	/* set usertag and progtag to what they should be */
	if (!(progtag = make_unicode("*VE NULL")))
		return ERROR_MALLOC;
	strcpy(buffer, ascii_user);
	strcat(buffer, " ");
	if (!(usertag = make_unicode(buffer)))
		return ERROR_MALLOC;

	/* open files */
	if ( !(source = fopen(input_filename, "rb")) )
	{
		fcloseall();
		return ERROR_OPEN_INPUT;
	}
	if ( !(dest = fopen(output_filename, "wb")) )
	{
		fcloseall();
		return ERROR_OPEN_INPUT;
	}

	/* this switch statement handles the directionality character, */
	/* whether present or absent. */
	switch (current = fgetuc(source))
	{
	case UERR:
		fcloseall();
		return ERROR_READ_INPUT;
	case 0xfeff:
		if (fputuc(0xfeff, dest) == UERR)
		{
			fcloseall();
			return ERROR_WRITE_OUTPUT;
		}
		break;
	default:
		if (fputuc(0xfeff, dest) == UERR)
		{
			fcloseall();
			return ERROR_WRITE_OUTPUT;
		}
		rewind(source);
		final = NONUNICODE;
	}


	while (!filedone)
	{
		wordlength = 0;

		/* iterate character loop */
		while (1)
		{
			previous = current;

			/* get a new character */
			if ((current = fgetuc(source)) == UERR)	/* end-of-file */
			{
				filedone = TRUE;
				resptag = usertag;
				break;
			}
			else if (wordlength == 0 && !whitespace(current))
			{
				/* write the initial line stuff now we know there's something to write */
				/* and assuming that this character is word initial */
				/* start of the line; initial line numbers inserted */
				sprintf(buffer, "s%05d w%03d ", sentences, words);

				if (unibuffer)
					free(unibuffer);
				if (!(unibuffer = make_unicode(buffer)))
				{
					fcloseall();
					return ERROR_MALLOC;
				}
				if (fputus(unibuffer, dest) == EOF)
				{
					fcloseall();
					return ERROR_WRITE_OUTPUT;
				}
			}

			if (whitespace(current))
			{
				if (whitespace(previous) || punctuation(previous) || previous == 0x003e)
					continue;
				else /* previous was a normal character */
				{
					resptag = usertag;
					break;
				}
			}
			else if (current == 0x003c)
			/* following block describes SGML element mode */
			{
				/* write the giveaway bracket */
				if (fputuc(current, dest) == UERR)
				{
					fcloseall();
					return ERROR_WRITE_OUTPUT;
				}
				wordlength++;
				while (1)
				{
					previous = current;
					if ((current = fgetuc(source)) == UERR)
					{
						filedone = TRUE;
						break;
					}
					if (fputuc(current, dest) == UERR)
					{
						fcloseall();
						return ERROR_WRITE_OUTPUT;
					}
					wordlength++;
					if (current == 0x003e)
						break;
				}
				resptag = progtag;
				words = 0;
				sentences++;
				break;
			}
			else /* the character in current is either a letter or a punctuation */
			{
				if (fputuc(current, dest) == UERR)
				{
					fcloseall();
					return ERROR_WRITE_OUTPUT;
				}
				wordlength++;

				/* conditions under which this is an end of word: prepared & evaluated */
				/* load peekahead, and if not EOF then rewind one character */
				if ((peekahead = fgetuc(source)) != UERR)
				{
					if (ungetuc(source))
					{
						fcloseall();
						return ERROR_READ_INPUT;
					}
				}
				else
					filedone = TRUE;
				if (punctuation(current) ||
					punctuation(peekahead) ||
					peekahead == 0x003c ||
					peekahead == UERR)
				{
					resptag = usertag;
					break;
				}
			}
		}
		/* end-of-the-word operations: */

		if (wordlength != 0)
		{
			buffer[0] = 0x09;
			buffer[1] = 0;
			if (unibuffer)
				free(unibuffer); 
			if (!(unibuffer = make_unicode(buffer)))
			{
				fcloseall();
				return ERROR_MALLOC;
			}
			if (fputus(unibuffer, dest) == EOF)
			{
				fcloseall();
				return ERROR_WRITE_OUTPUT;
			}

			/* write the resp tag and linebreak combination */
			if (fputus(resptag, dest) == EOF)
			{
				fcloseall();
				return ERROR_WRITE_OUTPUT;
			}
			if (fputus(linebreak, dest) == EOF)
			{
				fcloseall();
				return ERROR_WRITE_OUTPUT;
			}
			
			if (punctuation(current) && words > 3)
			{
				words = 0;
				sentences++;
			}
			words++;
		}
	}

	/* close files , free memory and return */
	if (fclose(source) != 0)
	{
		fcloseall();
		return ERROR_CLOSE_INPUT;
	}
	if (fclose(dest) != 0)
	{
		fcloseall();
		return ERROR_CLOSE_OUTPUT;
	}
	free(progtag);
	free(usertag);
	if (unibuffer)
		free(unibuffer);
	return final;
}