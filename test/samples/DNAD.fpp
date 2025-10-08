module dnad
  implicit none

  interface op1
    module procedure p1
  end interface

  interface op2
    module procedure p2
  end interface

  interface op3
    module procedure p3
  end interface

  interface op4
    module procedure p4
  end interface

  interface op5
    module procedure p5
  end interface

  interface op6
    module procedure p6
  end interface

  interface op7
    module procedure p7
  end interface

  interface op8
    module procedure p8
  end interface

  interface op9
    module procedure p9
  end interface

  interface op10
    module procedure p10
  end interface

  interface op11
    module procedure p11
  end interface

  interface op12
    module procedure p12
  end interface

  interface op13a
    module procedure p13
  end interface

  interface op13b
    module procedure p13
  end interface

  interface op14
    module procedure p14
  end interface

  interface op_abs
    module procedure abs_d
  end interface

  interface op_add
    module procedure add_di
  end interface

  interface op_assign
    module procedure assign_di
  end interface

contains
  function p0(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x
  end function p0

  function p1(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 1
  end function p1

  function p2(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 2
  end function p2

  function p3(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 3
  end function p3

  function p4(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 4
  end function p4

  function p5(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 5
  end function p5

  function p6(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 6
  end function p6

  function p7(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 7
  end function p7

  function p8(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 8
  end function p8

  function p9(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 9
  end function p9

  function p10(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 10
  end function p10

  function p11(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 11
  end function p11

  function p12(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 12
  end function p12

  function p13(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 13
  end function p13

  function p14(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 14
  end function p14

  function abs_d(x) result(y)
    real(8), intent(in) :: x
    real(8) :: y
    y = abs(x)
  end function abs_d

  subroutine add_di(x, i)
    real(8), intent(inout) :: x
    integer, intent(in) :: i
    x = x + real(i, kind=8)
  end subroutine add_di

  subroutine assign_di(x, i)
    real(8), intent(out) :: x
    integer, intent(in) :: i
    x = real(i, kind=8)
  end subroutine assign_di
end module dnad
