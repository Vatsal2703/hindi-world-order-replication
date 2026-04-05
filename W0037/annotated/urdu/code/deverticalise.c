#include <stdio.h>
#include "commandline.h"
#include "unicode.h"

int get_vert_line(unichar *buffer_word, unichar *buffer_tag, FILE *source);

main(int argc, char *argv[])
{
	FILE *source;
	FILE *dest;

	char *input_filename;
	char *output_filename;

	char wordsonline = 0;

	char go_on = 1;

	unichar thisword[200];
	unichar thistag[200];

	unichar buffer[800];

	unichar opentag1[] = { 0x003c, 0x0077, 0x0020, 0x0070, 0x006f, 0x0073, 0x003d, 0x0022, 0x0000 };
	unichar opentag2[] = { 0x0022, 0x003e, 0x0000 };
	unichar closetag[] = { 0x003c, 0x002f, 0x0077, 0x003e, 0x0020, 0x0000};
	unichar uninull[]  = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000};
	unichar uniPAR[]   = { 0x000d, 0x000a, 0x0000 };


	/* handle arguments */
	if (argc != 3)
	{
		cl_error_arguments(3, argc);
		return 1;
	}
	input_filename = argv[1];
	output_filename = argv[2];


	/* open files: binary */
	if( !(source = fopen(input_filename, "rb")) )
	{
		cl_error_file_open(input_filename);
		fcloseall();
		return 1;
	}
	if( !(dest = fopen(output_filename, "wb")) )
	{
		cl_error_file_open(output_filename);
		fcloseall();
		return 1;
	}
	/* insert directionality character */
	if ( fputuc( RIGHTWAY , dest) == UERR )
	{
		cl_error_file_write(output_filename);
		fcloseall();
		return 1;
	}
	/* skip directionality character */
	if ((buffer[0] = fgetuc(source)) == UERR)
	{
		cl_error_file_read(input_filename);
		fcloseall();
		return 1;
	}
	if (buffer[0] != RIGHTWAY)
		ungetuc(source);


	while (go_on == 1)
	{
		thisword[0] = 0;
		thistag[0] = 0;
		/* read a word and tag from a line of text */
		if (get_vert_line(thisword, thistag, source))
		{
			puts("End of file reached!");
			go_on = 0;
		}


		if (ustrident(thistag, uninull))
		{
			/* thisword is SGML tag, therefore write thisword, then follow with line break */
			if (fputus(thisword, dest) == EOF)
			{
				cl_error_file_write(output_filename);
				fcloseall();
				return 1;
			}

			/* write para break */
			if (fputus(uniPAR, dest) == EOF)
			{
				cl_error_file_write(output_filename);
				fcloseall();
				return 1;
			}
			wordsonline = 0;
		}
		else if (thisword[0])
		{
			/* assemble word / tag comb in buffer */

			ustrcpy(buffer, opentag1);
			ustrcat(buffer, thistag);
			ustrcat(buffer, opentag2);
			ustrcat(buffer, thisword);
			ustrcat(buffer, closetag);

			wordsonline++;

			/* line break after 5 words */
			if (wordsonline == 4)
			{
				ustrcat(buffer, uniPAR);
				wordsonline = 0;
			}

			/* write buffer */
			if (fputus(buffer, dest) == EOF)
			{
				cl_error_file_write(output_filename);
				fcloseall();
				return 1;
			}
		}
	}



	/* close both files */
	if (fclose(source) < 0)
	{
		cl_error_file_close(input_filename);
		fcloseall();
		return 1;
	}
	if (fclose(dest) < 0)
	{
		cl_error_file_close(output_filename);
		fcloseall();
		return 1;
	}
	
	return 0;

}




int get_vert_line(unichar *buffer_word, unichar *buffer_tag, FILE *source)
{
	int i;
	unichar uc;

	/* scroll past the first 12 characters of the line */

	for ( i = 0 ; i < 12 ; i++ )
	{
		if ((uc = fgetuc(source)) == UERR)
			return 1;
	}


	/* load word until char == tab 0x0009 */

	for ( i = 0 ; 1 ; i++ )
	{
		if ((uc = fgetuc(source)) == UERR)
			return 1;

		if (uc == 0x0009)
		{
			*(buffer_word+i) = 0x0000;
			break;
		}
		else
			*(buffer_word+i) = uc;
	}

	/* skip another 4 chars */

	for ( i = 0 ; i < 4 ; i++ )
	{
		if ((uc = fgetuc(source)) == UERR)
			return 1;
	}

	/* fgetus everything until end of line as tag */
	if (fgetus(buffer_tag, source) == NULL)
		return 1;

	/* this bit wipes away any whitespace hanging around after the tag */
	for ( i = 0 ; *(buffer_tag+i) ; i++)
	{
		if ( whitespace(*(buffer_tag+i)) )
		{
			*(buffer_tag+i);
			break;
		}
	}

	return 0;
}