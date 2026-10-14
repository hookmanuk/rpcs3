#include "shortcut_dialog.h"
#include "ui_shortcut_dialog.h"
#include "shortcut_settings.h"
#include "gui_settings.h"

#include <QDialogButtonBox>
#include <QPushButton>
#include <QKeySequenceEdit>
#include <QLabel>
#include <QHBoxLayout>
#include <QScreen> // VR fork
#include <QScrollBar> // VR fork

shortcut_dialog::shortcut_dialog(const std::shared_ptr<gui_settings> gui_settings, QWidget* parent)
	: QDialog(parent), ui(new Ui::shortcut_dialog), m_gui_settings(gui_settings)
{
	ui->setupUi(this);

	connect(ui->buttonBox, &QDialogButtonBox::rejected, this, &shortcut_dialog::reject);
	connect(ui->buttonBox, &QDialogButtonBox::clicked, [this](QAbstractButton* button)
	{
		if (button == ui->buttonBox->button(QDialogButtonBox::Save))
		{
			save();
			accept();
		}
		else if (button == ui->buttonBox->button(QDialogButtonBox::Apply))
		{
			save();
		}
	});

	shortcut_settings sc_settings{};

	for (const auto& [shortcut_key, shortcut] : sc_settings.shortcut_map)
	{
		const QKeySequence key_sequence = sc_settings.get_key_sequence(shortcut, gui_settings);

		QLabel* label = new QLabel(shortcut.localized_name);

		QKeySequenceEdit* key_sequence_edit = new QKeySequenceEdit;
		key_sequence_edit->setObjectName(shortcut.name);
		key_sequence_edit->setMinimumWidth(label->sizeHint().width());
		key_sequence_edit->setKeySequence(key_sequence);
		key_sequence_edit->setClearButtonEnabled(true);

		m_values[shortcut.name] = key_sequence.toString();

		connect(key_sequence_edit, &QKeySequenceEdit::keySequenceChanged, this, &shortcut_dialog::handle_change);

		QHBoxLayout* shortcut_layout = new QHBoxLayout;
		shortcut_layout->addWidget(label);
		shortcut_layout->addWidget(key_sequence_edit);
		shortcut_layout->setStretch(0, 1);
		shortcut_layout->setStretch(1, 1);

		const auto add_layout = [](QVBoxLayout* layout, QHBoxLayout* shortcut_layout)
		{
			layout->insertLayout(layout->count() - 1, shortcut_layout); // count() - 1 to ignore the vertical spacer
		};

		switch (shortcut.handler_id)
		{
		case gui::shortcuts::shortcut_handler_id::game_window:
		{
			add_layout(ui->game_window_layout, shortcut_layout);
			break;
		}
		case gui::shortcuts::shortcut_handler_id::main_window:
		{
			add_layout(ui->main_window_layout, shortcut_layout);
			break;
		}
		}
	}

	const int min_width = std::max(
	{
		ui->main_window_group_box->sizeHint().width(),
		ui->game_window_group_box->sizeHint().width(),
	});

	ui->main_window_group_box->setMinimumWidth(min_width);
	ui->game_window_group_box->setMinimumWidth(min_width);
	vr_fit_to_screen(); // VR fork: the lists scroll; full height where the screen has room
}

void shortcut_dialog::vr_fit_to_screen()
{
	// VR fork. Both columns side by side without a horizontal scroll bar, and as tall as the lists up to 90 % of the
	// screen (the game window list outgrew small screens).
	ui->scroll_area->setMinimumWidth(ui->scroll_contents->minimumSizeHint().width() + ui->scroll_area->verticalScrollBar()->sizeHint().width());
	const QMargins m = layout()->contentsMargins();
	const int full = ui->scroll_contents->sizeHint().height() + ui->buttonBox->sizeHint().height() + layout()->spacing() + m.top() + m.bottom();
	const QScreen* s = screen();
	resize(width(), s ? std::min(full, s->availableGeometry().height() * 9 / 10) : full);
}

shortcut_dialog::~shortcut_dialog()
{
	delete ui;
}

void shortcut_dialog::save()
{
	shortcut_settings sc_settings{};

	for (const auto& entry : m_values)
	{
		m_gui_settings->SetValue(sc_settings.get_shortcut_gui_save(entry.first), entry.second, false);
	}
	m_gui_settings->sync();

	Q_EMIT saved();
}

void shortcut_dialog::handle_change(const QKeySequence& keySequence)
{
	m_values[sender()->objectName()] = keySequence.toString();
}
