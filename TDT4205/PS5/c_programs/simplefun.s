.globl	main
.text
main:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	$0, %eax
	call	funcall
	movl	$0, %eax
	popq	%rbp
	ret
funcall:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$16, %rsp
	movl	$10, %esi
	movl	$5, %edi
	movl	$0, %eax
	call	my_function
	movl	%eax, -4(%rbp)
	leave
	ret
.LC0:
	.string	"Parameter s is %d, t is %d\n"
	.text
my_function:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-8(%rbp), %edx
	movl	-4(%rbp), %eax
	movl	%eax, %esi
	movl	$.LC0, %edi
	movl	$0, %eax
	call	printf
	movl	$0, %eax
	leave
	ret