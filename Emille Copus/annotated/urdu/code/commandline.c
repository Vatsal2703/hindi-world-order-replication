#include <stdio.h>
#include "andrew.h"
#include "commandline.h"

void cl_enter_to_continue(void)
{
	char junk[80];
	fputs("Press [ENTER] to continue :  ", stdout);
	gets(junk);
}


int cl_query(const char *string)
{
	puts(string);
	return (input_yes_no());
}


void cl_error_arguments(int required, int got)
{
	puts("You have supplied too few arguments.");

	printf("This program requires %d arguments after the program name.\n", required-1);
	if (required > got)
		printf("You require an additional %d arguments.\n", required-got);
	else
		printf("You have supplied %d too many arguments.\n", got-required);
}


/* Note: the error functions at the command line do not auto-abort. */
/* This should be done by the calling function. */

void cl_error_file_close(const char *file)
{	fprintf(stderr, "There was an error closing %s. Output may contain errors.\n", file);	}


void cl_error_file_open(const char *file)
{	fprintf(stderr, "There was an error opening %s. Output may contain errors.\n", file);	}


void cl_error_file_read(const char *file)
{	fprintf(stderr, "There was an error reading from %s. Output may contain errors.\n", file);	}


void cl_error_file_write(const char *file)
{	fprintf(stderr, "There was an error writing to %s. Output may contain errors.\n", file);	}


void cl_error_memory_alloc(void)
{	fputs("There has been a memory allocation error. Output may contain errors.\n", stderr);	}



/* returns 0 if file doesn't exist or user chooses to overwrite, 1 if the file exists	*/
int cl_test_file_write(const char *filename)
{
	FILE *file_test;

	if (file_test = fopen(filename, "rb"))
	{
		fclose(file_test);
		
		printf("File %s already exists.\n", filename);
		puts("You will lose all previous data in that file if you continue.");
		
		fputs("Do you wish to overwrite the file?  ", stdout);
		if (input_yes_no() == NO)
		{
			puts(" ");
			return 1;
		}
	}
	return 0;
}
